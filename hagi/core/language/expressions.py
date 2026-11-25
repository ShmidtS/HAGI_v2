"""Дерево выражения, программа, векторизованный вычислитель — модуль символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

from dataclasses import dataclass  # импорт стандартной библиотеки, нужный этой части
from typing import TYPE_CHECKING, Any, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.memory.aliases import BINARY_OPS, TOKEN_BITS, VAR_NAMES  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.synthesis.prefilter import OPS  # имена из «prefilter» (прежний общий импорт из пакета наследия)
from hagi.core.memory.aliases import gamma_bits, zigzag  # имена из «aliases» (прежний общий импорт из пакета наследия)
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.growth import Expert  # класс эксперта из части «growth»: имя нужно лишь в строковых аннотациях

# ==============================================================================
# 3. Дерево выражения, программа и векторизованный вычислитель
# ==============================================================================

# Реестр операторов, построенных методом переификации: изобретённые координаты
# становятся полноправными унарными и бинарными макрооперациями внутри
# выражений Expr. Заполняется в hagi.invent.representations функцией reify_construct();
REIFIED_OPS: dict = {}  # реестр макрооператоров: имя -> функция


@dataclass(frozen=True)  # неизменяемый класс: экземпляры нельзя менять после создания
class Expr:
    """Узел дерева выражения (направленный ациклический граф вызовов): операция и её аргументы.

    op — имя операции; args — кортеж подвыражений-аргументов; value —
    дополнительное значение: индекс переменной, константа, имя вызываемого
    эксперта, коэффициенты многочлена, множество или список диапазонов."""
    op: str  # имя операции
    args: tuple["Expr", ...] = ()  # подвыражения-аргументы; по умолчанию пусто
    value: Optional[Any] = None  # дополнительное значение узла; по умолчанию нет

    @staticmethod  # метод без доступа к экземпляру
    def var(i: int) -> "Expr":
        """Построить узел переменной: i-й аргумент входа."""
        return Expr("var", (), int(i))  # узел var с индексом i

    @staticmethod  # метод без доступа к экземпляру
    def const(v: int) -> "Expr":
        """Построить узел константы со значением v."""
        return Expr("const", (), int(v))  # узел const со значением v

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "Expert"]] = None  # аргументы: вход xs и среда env (по умолчанию нет)
    ) -> int:  # возвращает целое значение
        """Вычислить выражение на входе xs: значение int.

        xs — входные аргументы; env — словарь имени эксперта -> объект
        эксперта для операции call (необязателен). Операции:
        var: xs[value]; const: value; call: программа вызываемого эксперта
        на значениях аргументов; anf: XOR коэффициентов c мономов s ⊆ mask_x,
        где mask_x = Σ b_i·2^(n−1−i); if_cond: вторая ветвь, если условие
        истинно, иначе третья; poly: схема Горнера acc = acc·x + c_i по
        коэффициентам от старшего к младшему; poly_nd: Σ coef·Π x_j^p_j;
        далее базовые операции и операторы из REIFIED_OPS и OPS."""
        if self.op == "var":  # операция — переменная
            return int(xs[int(self.value)])  # взять аргумент xs[value]
        if self.op == "const":  # операция — константа
            return int(self.value)  # вернуть значение value
        if self.op == "call":  # операция — вызов другого эксперта
            if env is None or self.value not in env:  # нет среды или вызываемый эксперт в ней не зарегистрирован
                raise ValueError(f"missing called expert: {self.value}")  # ошибка: эксперт отсутствует
            vals = tuple(a.eval(xs, env) for a in self.args)  # значения аргументов вызова (рекурсивно)
            callee = env[self.value]  # вызываемый эксперт по имени
            if not callee.supports(vals):  # вход вне проверенной области применимости эксперта
                raise ValueError(  # ошибка: вызов вне сертифицированной области
                    f"call to {self.value} outside certified support: {vals}"  # текст ошибки с именем и аргументами
                )
            return callee.program.eval(vals, env)  # результат программы вызываемого эксперта на значениях аргументов
        if self.op == "anf":  # операция — алгебраическая нормальная форма (АНФ)
            n_vars, coeffs = self.value  # число переменных n и список пар (моном s, коэффициент c)
            mask_x = 0  # битовая маска входа, изначально 0
            for i in range(n_vars):  # по номерам переменных i = 0..n−1
                b = int(xs[i])  # значение i-й переменной
                if b not in (0, 1):  # вход должен быть булевым
                    raise ValueError("anf expects boolean inputs")  # ошибка: АНФ принимает только булевы входы
                if b:  # если b_i = 1
                    mask_x |= 1 << (n_vars - 1 - i)  # установить бит n−1−i: mask_x |= 2^(n−1−i)
            acc = 0  # аккумулятор результата, изначально 0
            for s, c in coeffs:  # по парам (моном s, коэффициент c)
                if (s & mask_x) == s:  # моном входит в вход: s AND mask_x = s
                    acc ^= c  # аккумулятор XOR коэффициент
            return int(acc)  # значение АНФ
        if self.op == "if_cond":  # операция — условное выражение
            cond_val = self.args[0].eval(xs, env)  # значение условия (первый аргумент)
            return (  # вернуть вторую ветвь, если условие истинно, иначе третью
                self.args[1].eval(xs, env) if cond_val else self.args[2].eval(xs, env)  # ветвь «то» args[1] при cond_val ≠ 0, иначе ветвь «иначе» args[2]
            )
        vals = [a.eval(xs, env) for a in self.args]  # значения всех аргументов (рекурсивно)
        if self.op == "poly":  # операция — многочлен одного аргумента
            acc = 0  # аккумулятор схемы Горнера, изначально 0
            for ci in reversed(list(self.value or ())):  # по коэффициентам от старшего к младшему (пустой список, если value нет)
                acc = acc * vals[0] + int(ci)  # шаг Горнера: acc = acc·x + c_i
            return int(acc)  # значение многочлена
        if self.op == "poly_nd":  # операция — многочлен многих переменных
            total = 0  # сумма членов, изначально 0
            for coef, powers in self.value or ():  # по парам (коэффициент, показатели степени)
                term = int(coef)  # член начинается с коэффициента
                for x_val, p in zip(xs, powers):  # по парам (переменная x_j, показатель p_j)
                    term *= int(x_val) ** int(p)  # умножить член: term = term·x_j^p_j
                total += term  # прибавить член: total = total + term
            return int(total)  # сумма членов total
        if self.op == "absdiff":  # операция — абсолютная разность
            return abs(vals[0] - vals[1])  # вернуть |a_0 − a_1|
        if self.op == "fma":  # операция — умножение со сложением
            return vals[0] * vals[1] + vals[2]  # вернуть a_0·a_1 + a_2
        if self.op == "clamp":  # операция — ограничение диапазоном
            return min(max(vals[0], vals[1]), vals[2])  # вернуть min(max(a_0, a_1), a_2)
        if self.op == "median3":  # операция — медиана трёх
            return sorted(vals)[1]  # средний элемент отсортированных аргументов
        if self.op == "clamp3":  # операция — ограничение отрезком [−3, 3]
            return min(max(vals[0], -3), 3)  # вернуть min(max(a_0, −3), 3)
        if self.op == "mul3":  # операция — произведение трёх
            return vals[0] * vals[1] * vals[2]  # вернуть a_0·a_1·a_2
        if self.op == "in_set":  # операция — принадлежность конечному множеству
            # значение a_0 должно входить в строго возрастающее конечное множество;
            # разреженное запоминание ответов разрешено только для задач
            # языкового семейства (проверяется при занесении в библиотеку),
            # но не как общий способ находить законы.
            return int(vals[0] in set(self.value or ()))  # 1, если a_0 ∈ value, иначе 0
        if self.op == "in_ranges":  # операция — принадлежность диапазонам
            # включающая принадлежность отсортированным непересекающимся
            # несмежным отрезкам — парам (lo, hi).
            v = vals[0]  # проверяемое значение v = a_0
            return int(any(lo <= v <= hi for lo, hi in (self.value or ())))  # 1, если найдётся отрезок с lo ≤ v ≤ hi, иначе 0
        if self.op == "construct":  # операция — конструкция из реестра
            from hagi.invent.representations import construct_eval  # вычислитель конструкций импортируется здесь, чтобы избежать циклического импорта

            return construct_eval(self.value, xs, env)  # результат конструкции на входе xs в среде env
        fn = REIFIED_OPS.get(self.op)  # функция макрооператора из реестра, если есть
        if fn is not None:  # оператор найден в реестре
            return int(fn(*vals))  # применить его к значениям аргументов
        fn = OPS.get(self.op)  # иначе — базовая операция из OPS
        if fn is None:  # операция неизвестна
            raise ValueError(f"unsupported op: {self.op}")  # ошибка: операция не поддерживается
        return int(fn(*vals))  # применить базовую операцию к значениям аргументов

    @property  # сделать метод свойством: вызывается без скобок
    def cost(self) -> int:
        """Стоимость выражения cost: число узлов дерева (1 + сумма стоимостей аргументов).

        anf: 1 + число мономов; poly и poly_nd: 1 + число коэффициентов
        (членов); construct: 1 + число бит конструкции. Результат запоминается
        в атрибуте _cost_memo, поэтому повторный вызов не пересчитывает."""
        c = self.__dict__.get("_cost_memo")  # взять запомненную стоимость, если она уже вычислялась
        if c is None:  # стоимость ещё не вычислена
            if self.op in ("var", "const"):  # переменная или константа
                c = 1  # лист дерева имеет стоимость 1
            elif self.op == "anf":  # алгебраическая нормальная форма
                _, coeffs = self.value  # извлечь список пар (моном, коэффициент); число переменных не нужно
                c = 1 + len(coeffs)  # стоимость = 1 + число мономов
            elif self.op in ("poly", "poly_nd"):  # многочлен одного или многих аргументов
                c = 1 + len(self.value or ())  # стоимость = 1 + число коэффициентов (0, если value пусто)
            elif self.op == "construct":  # конструкция из реестра
                from hagi.invent.representations import construct_bits as _cb  # импортировать здесь оценку числа бит конструкции, чтобы избежать циклического импорта

                c = 1 + _cb(self.value)  # стоимость = 1 + битовая длина конструкции
            else:  # любая другая операция
                c = 1 + sum(a.cost for a in self.args)  # стоимость = 1 + Σ cost(a_i) по аргументам
            object.__setattr__(self, "_cost_memo", c)  # запомнить стоимость, обойдя запрет изменения замороженного объекта
        return c  # вернуть стоимость выражения

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        """Число бит описания выражения: код токена плюс полезная нагрузка и биты аргументов.

        expert_index — словарь «имя эксперта -> номер», нужный операции call.
        Везде TOKEN_BITS — цена кода операции; gamma_bits(n) = 2·(⌊log2 n⌋) + 1
        — код Элиаса-гамма числа n ≥ 1; zigzag(v) = 2v при v ≥ 0 и −2v − 1 при
        v < 0 — перевод целого в неотрицательное число."""
        if self.op == "var":  # переменная
            return TOKEN_BITS + gamma_bits(int(self.value or 0) + 1)  # var: TOKEN_BITS + gamma_bits(индекс + 1)
        if self.op == "const":  # константа
            return TOKEN_BITS + gamma_bits(zigzag(int(self.value)) + 1)  # const: TOKEN_BITS + gamma_bits(zigzag(значение) + 1)
        if self.op == "call":  # вызов другого эксперта
            idx = expert_index.get(str(self.value), 0) if expert_index else 0  # номер вызываемого эксперта (0, если словарь не задан или имени нет)
            return (  # итог: call: TOKEN_BITS
                TOKEN_BITS  # плюс код токена операции
                + gamma_bits(idx + 1)  # плюс gamma_bits(номер эксперта + 1)
                + sum(a.bits(expert_index) for a in self.args)  # плюс сумма бит аргументов
            )
        # та же формула дельта-кодирования множества и диапазонов, что и в
        # hagi.certificates._expr_bits: без неё стоимость запоминания ответов
        # считается заниженной.
        if self.op == "in_set":  # принадлежность конечному множеству
            pts = [int(v) for v in (self.value or ())]  # список точек множества как int
            if not pts:  # множество пусто
                return TOKEN_BITS  # пустое множество стоит только кода токена
            pay = (gamma_bits(len(pts) + 1)  # нагрузка = gamma_bits(число точек + 1)
                   + gamma_bits(zigzag(pts[0]) + 1))  # плюс gamma_bits(zigzag(первая точка) + 1)
            pay += sum(gamma_bits(b - a) for a, b in zip(pts, pts[1:]))  # плюс Σ gamma_bits(p_{k+1} − p_k) по соседним точкам — дельта-кодирование
            return TOKEN_BITS + pay + sum(a.bits(expert_index)  # вернуть TOKEN_BITS + нагрузка + биты аргументов
                                          for a in self.args)  # сумма бит аргументов
        if self.op == "in_ranges":  # принадлежность диапазонам
            rs = [(int(lo), int(hi)) for lo, hi in (self.value or ())]  # список пар (lo, hi) как int
            if not rs:  # список пуст
                return TOKEN_BITS  # нет диапазонов — только код токена
            pay = (gamma_bits(len(rs) + 1)  # нагрузка = gamma_bits(число диапазонов + 1)
                   + gamma_bits(zigzag(rs[0][0]) + 1)  # плюс gamma_bits(zigzag(lo первого диапазона) + 1)
                   + gamma_bits(rs[0][1] - rs[0][0] + 1))  # плюс gamma_bits(длина первого диапазона hi − lo + 1)
            for pr, c in zip(rs, rs[1:]):  # по парам (предыдущий pr, текущий c) диапазонов
                pay += (gamma_bits(c[0] - pr[1])  # прибавить gamma_bits(промежуток c.lo − pr.hi)
                        + gamma_bits(c[1] - c[0] + 1))  # плюс gamma_bits(длина c.hi − c.lo + 1)
            return TOKEN_BITS + pay + sum(a.bits(expert_index)  # вернуть TOKEN_BITS + нагрузка + биты аргументов
                                          for a in self.args)  # сумма бит аргументов
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список пар (моном s, коэффициент c)
            payload = gamma_bits(n_vars + 1) + gamma_bits(len(coeffs) + 1)  # нагрузка = gamma_bits(n + 1) + gamma_bits(число мономов + 1)
            payload += sum(gamma_bits(s + 1) for s, _ in coeffs)  # плюс Σ gamma_bits(s + 1) по мономам
            return TOKEN_BITS + payload  # вернуть TOKEN_BITS + нагрузка
        if self.op == "poly":  # многочлен одного аргумента
            coeffs = tuple(int(v) for v in (self.value or ()))  # кортеж коэффициентов как int
            payload = gamma_bits(len(coeffs) + 1) + sum(  # нагрузка = gamma_bits(число коэффициентов + 1) + сумма
                gamma_bits(zigzag(v) + 1) for v in coeffs  # Σ gamma_bits(zigzag(v) + 1) по коэффициентам v
            )
            return TOKEN_BITS + payload + sum(a.bits(expert_index) for a in self.args)  # вернуть TOKEN_BITS + нагрузка + сумма бит аргументов
        if self.op == "poly_nd":  # многочлен многих переменных
            terms = tuple(self.value or ())  # кортеж членов (коэффициент, показатели степени)
            payload = gamma_bits(len(terms) + 1)  # нагрузка = gamma_bits(число членов + 1)
            for c, powers in terms:  # по членам (коэффициент c, показатели powers)
                payload += gamma_bits(zigzag(int(c)) + 1) + gamma_bits(len(powers) + 1)  # прибавить gamma_bits(zigzag(c) + 1) + gamma_bits(число показателей + 1)
                payload += sum(gamma_bits(int(p) + 1) for p in powers)  # прибавить Σ gamma_bits(p + 1) по показателям p
            return TOKEN_BITS + payload  # вернуть TOKEN_BITS + нагрузка
        return TOKEN_BITS + sum(a.bits(expert_index) for a in self.args)  # прочие операции: TOKEN_BITS + сумма бит аргументов

    def callees(self) -> set[str]:
        """Множество имён экспертов, вызываемых в этом выражении и его подвыражениях."""
        out: set[str] = set()  # накопитель имён
        if self.op == "call" and self.value is not None:  # узел — вызов и имя задано
            out.add(str(self.value))  # добавить имя вызываемого эксперта
        for a in self.args:  # по аргументам
            out |= a.callees()  # объединить с именами аргумента
        return out  # вернуть множество имён

    def substitute(self, mapping: dict[int, "Expr"]) -> "Expr":
        """Подставить в выражение новые подвыражения вместо переменных: новое выражение Expr.

        mapping — словарь «индекс переменной -> выражение»; переменные, которых
        нет в словаре, и узлы без аргументов остаются как есть."""
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # индекс переменной
            return mapping.get(idx, self)  # вернуть замену из словаря или сам узел, если замены нет
        if not self.args:  # узел без аргументов
            return self  # не меняется
        return Expr(  # собрать новый узел
            self.op, tuple(a.substitute(mapping) for a in self.args), self.value  # с теми же операцией и значением, но с рекурсивно заменёнными аргументами
        )

    def subtrees(self) -> list["Expr"]:
        """Список всех поддеревьев выражения, включая само выражение (корень первым)."""
        out = [self]  # начать с самого узла
        for a in self.args:  # по аргументам
            out.extend(a.subtrees())  # добавить все их поддеревья
        return out  # список поддеревьев

    def replace_subtree(self, target: "Expr", replacement: "Expr") -> "Expr":
        """Заменить в дереве поддерево target на replacement: новое выражение Expr.

        Сравнение — по равенству значений (узлы заморожены и сравнимы как
        кортежи полей)."""
        if self == target:  # текущий узел совпадает с искомым
            return replacement  # вернуть замену
        if not self.args:  # узел без аргументов
            return self  # остаётся как есть
        return Expr(  # собрать новый узел
            self.op,  # с той же операцией
            tuple(a.replace_subtree(target, replacement) for a in self.args),  # с рекурсивно обработанными аргументами
            self.value,  # и тем же значением
        )

    def pretty(self) -> str:
        """Читаемая запись выражения строкой: имена переменных, знаки операций, скобки."""
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # индекс переменной
            return VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя из VAR_NAMES, если индекс в пределах списка, иначе x<индекс>
        if self.op == "const":  # константа
            return str(self.value)  # запись числа
        if self.op == "call":  # вызов эксперта
            return f"{self.value}({', '.join(a.pretty() for a in self.args)})"  # «имя(аргумент через запятую)»
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список мономов с коэффициентами
            terms = []  # список текстов мономов
            for s, _ in coeffs:  # по мономам s (коэффициент не входит в текст)
                if s == 0:  # пустой моном
                    terms.append("1")  # записывается как константа 1
                else:  # иначе
                    vs = [  # список имён переменных монома
                        VAR_NAMES[i] if i < len(VAR_NAMES) else f"x{i}"  # имя переменной i, если i в пределах VAR_NAMES, иначе x<i>
                        for i in range(n_vars)  # по i = 0..n−1
                        if (s >> (n_vars - 1 - i)) & 1  # брать только переменные, чей бит (s >> (n−1−i)) & 1 установлен
                    ]  # закрытие списка имён
                    terms.append("&".join(vs))  # моном — имена переменных через «&»
            return "anf(" + (" ^ ".join(terms) if terms else "0") + ")"  # итог: «anf(мономы через ^)», а при пустом списке «anf(0)»
        if self.op == "if_cond":  # условное выражение
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"  # «(то если условие иначе иначе)»
        if self.op == "poly":  # многочлен одного аргумента
            coeffs = tuple(self.value or ())  # кортеж коэффициентов от младшего к старшему
            terms = []  # список текстов членов
            for i in range(len(coeffs) - 1, -1, -1):  # по степеням i от старшей к младшей
                c = int(coeffs[i])  # коэффициент при x^i
                if c == 0:  # нулевой коэффициент
                    continue  # не показывать
                mag = abs(c)  # модуль коэффициента
                body = "" if i == 0 else ("x" if i == 1 else f"x**{i}")  # множитель x^i: пусто при i = 0, x при i = 1, иначе x**i
                term = str(mag) if i == 0 else (body if mag == 1 else f"{mag}*{body}")  # текст члена: модуль при i = 0, иначе множитель или «модуль*множитель»
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + term)  # знак минус перед ним ставится, только если коэффициент отрицателен
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + term)  # перед ними « - » или « + » по знаку коэффициента
            return "poly(" + ("".join(terms) if terms else "0") + ")"  # итог: «poly(члены)», а при пустом списке «poly(0)»
        if self.op == "poly_nd":  # многочлен многих переменных
            terms = []  # список текстов членов
            for coef, powers in sorted(  # по членам, отсортированным по убыванию полной степени и лексикографически
                (self.value or ()), key=lambda it: (sum(it[1]), it[1]), reverse=True  # ключ сортировки: (сумма показателей, вектор показателей)
            ):
                c = int(coef)  # коэффициент члена
                if c == 0:  # нулевой коэффициент
                    continue  # не показывать
                factors = []  # список множителей x_j^p_j
                for idx, p in enumerate(powers):  # по номерам переменных idx и показателям p
                    p = int(p)  # показатель как int
                    if p == 0:  # показатель 0
                        continue  # переменная не входит в член
                    nm = VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя переменной
                    factors.append(f"{nm}^{p}" if p != 1 else nm)  # множитель «имя^p» при p ≠ 1, иначе просто имя
                body = "*".join(factors) if factors else str(abs(c))  # тело члена: произведение множителей, а без множителей — модуль коэффициента
                if factors and abs(c) != 1:  # есть множители и модуль коэффициента не равен 1
                    body = f"{abs(c)}*{body}"  # дописать модуль коэффициента перед произведением
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + body)  # минус перед ним, только если коэффициент отрицателен
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + body)  # перед ними « - » или « + » по знаку коэффициента
            return "poly_nd(" + ("".join(terms) if terms else "0") + ")"  # итог: «poly_nd(члены)», а при пустом списке «poly_nd(0)»
        if self.op == "clamp3":  # ограничение отрезком [−3, 3]
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # «min(max(аргумент, -3), 3)»
        if self.op == "absdiff":  # абсолютная разность
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # «abs(первый − второй)»
        if self.op == "fma":  # умножение со сложением
            return f"({self.args[0].pretty()} * {self.args[1].pretty()} + {self.args[2].pretty()})"  # «(первый * второй + третий)»
        if self.op == "clamp":  # ограничение диапазоном
            return f"min(max({self.args[0].pretty()}, {self.args[1].pretty()}), {self.args[2].pretty()})"  # «min(max(значение, нижняя граница), верхняя граница)»
        if self.op == "median3":  # медиана трёх
            return f"median({', '.join(a.pretty() for a in self.args)})"  # «median(аргументы через запятую)»
        if self.op == "mul3":  # произведение трёх
            return " * ".join(a.pretty() for a in self.args)  # аргументы через « * »
        if self.op == "in_set":  # принадлежность конечному множеству
            return f"in_set({sorted(set(self.value or ()))})"  # «in_set(отсортированное множество)»
        if self.op == "in_ranges":  # принадлежность диапазонам
            return f"in_ranges({self.value})"  # «in_ranges(список диапазонов)»
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # «операция(аргумент)»
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # «(первый операция второй)» — инфиксная запись
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # иначе «операция(все аргументы через запятую)»

    def to_dict(self) -> dict[str, Any]:
        """Сериализация выражения в словарь для записи в JSON."""
        val = self.value  # значение узла
        if self.op == "anf" and val is not None:  # АЛН со значением
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]  # value = [n, [[s, c], …]] — списки вместо кортежей
        elif self.op == "construct" and isinstance(val, dict):  # конструкция, заданная словарём
            from hagi.invent.representations import construct_to_dict_safe  # импортируется здесь, чтобы избежать циклического импорта

            val = construct_to_dict_safe(val)  # преобразовать словарь конструкции в безопасный для JSON вид
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}  # словарь «имя операции, значение, список аргументов»

    @staticmethod  # метод без доступа к экземпляру — восстановление
    def from_dict(d: dict[str, Any]) -> "Expr":  # имя операции
        """Восстановить выражение Expr из словаря, полученного to_dict."""
        op = str(d["op"])  # значение узла или None
        val = d.get("value")  # многочлен одного аргумента
        if op == "poly" and val is not None:  # коэффициенты — кортеж int
            val = tuple(int(v) for v in val)  # многочлен многих переменных
        elif op == "poly_nd" and val is not None:  # value = ((коэффициент, (показатели)), …) — кортежи int
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)  # АЛН
        elif op == "anf" and val is not None:  # value = (n, ((моном, коэффициент), …)) — кортежи int
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))  # конструкция, заданная словарём
        elif op == "construct" and isinstance(val, dict):  # импортируется здесь, чтобы избежать циклического импорта
            from hagi.invent.representations import construct_from_dict_safe  # импортируется здесь, чтобы избежать циклического импорта
  # восстановить значение конструкции из словаря
            val = construct_from_dict_safe(val)  # новый узел с рекурсивно восстановленными аргументами
        return Expr(op, tuple(Expr.from_dict(a) for a in d.get("args", [])), val)  # новый узел: операция, аргументы из словаря, значение


@dataclass(frozen=True)  # неизменяемый класс
class Program:
    """Программа эксперта: либо выражение Expr, либо булева диаграмма Венна по маске.

    arity — число аргументов; expr — дерево выражения (для обычных программ);
    venn_mask — 4-битная таблица истинности булевой функции двух аргументов
    f(x, y) = (venn_mask >> (2·x + y)) & 1."""
    arity: int  # число аргументов
    expr: Optional[Expr] = None  # дерево выражения; по умолчанию нет
    venn_mask: Optional[int] = None  # маска булевой функции; по умолчанию нет

    @property  # сделать метод свойством
    def cost(self) -> int:
        """Стоимость программы cost: 4 для диаграммы Венна, иначе стоимость выражения."""
        return (  # вернуть
            4  # вернуть стоимость: 4,
            if self.venn_mask is not None  # если задана маска Венна,
            else (self.expr.cost if self.expr else 10**9)  # иначе cost выражения, а без выражения — 10^9 (бесконечно дорого)
        )

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        """Число бит описания программы: TOKEN_BITS + 4 для Венна, иначе биты выражения."""
        if self.venn_mask is not None:  # задана маска Венна
            return TOKEN_BITS + 4  # TOKEN_BITS + 4 бита маски
        if self.expr is None:  # выражения нет
            return 10**9  # 10^9 — бесконечно большое описание
        return self.expr.bits(expert_index)  # биты выражения

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "Expert"]] = None  # аргументы: вход xs и среда env (по умолчанию нет)
    ) -> int:  # возвращает значение int
        """Вычислить программу на входе xs: значение int или ошибка.

        Проверяет число аргументов; для Венна — два булевых входа; иначе
        вычисляет выражение. env — среда вызываемых экспертов."""
        if len(xs) != self.arity:  # число аргументов должно равняться арности
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка о неверном числе входов
        if self.venn_mask is not None:  # программа — диаграмма Венна
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # Венн требует арность 2 и значения x, y ∈ {0, 1}
                raise ValueError("BoolVenn expects two binary inputs")  # ошибка: ожидались два булевых входа
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1  # f(x, y) = (маска >> (2·x + y)) & 1
        if self.expr is None:  # выражения нет
            raise RuntimeError("empty program")  # ошибка: пустая программа
        return self.expr.eval(xs, env)  # значение выражения на входе

    def pretty(self) -> str:
        """Читаемая запись программы строкой."""
        if self.venn_mask is not None:  # диаграмма Венна
            return f"BoolVenn(mask={self.venn_mask:04b})"  # «BoolVenn(mask=4 бита маски)»
        assert self.expr is not None  # далее выражение обязано существовать
        return self.expr.pretty()  # читаемая запись выражения

    def to_dict(self) -> dict[str, Any]:
        """Сериализация программы в словарь для JSON: арность, маска, выражение."""
        return {  # словарь
            "arity": self.arity,  # арность
            "venn_mask": self.venn_mask,  # маска Венна или None
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения или None
        }

    @staticmethod  # метод без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "Program":
        """Восстановить программу Program из словаря, полученного to_dict."""
        return Program(  # новая программа
            int(d["arity"]),  # с арностью
            Expr.from_dict(d["expr"]) if d.get("expr") else None,  # выражением из словаря (или None, если его нет)
            d.get("venn_mask"),  # и маской Венна (или None)
        )


_VEC_MAG_LIMIT = 1 << 28  # предел модуля числа для векторизованных вычислений: 2^28, чтобы произведение не переполнило int64


def eval_expr_vec(
    expr: Expr, X: np.ndarray, env: Optional[dict[str, "Expert"]] = None  # аргументы: выражение, матрица входов, среда env
) -> tuple[np.ndarray, np.ndarray]:  # возвращает кортеж (значения, флаги)
    """Векторизованный вычислитель выражения на матрице входов X: (значения, флаги ok).

    X — (N, число переменных) int64; env — среда вызываемых экспертов.
    Возвращает значения (N,) int64 и ok (N,) bool. Вычисление осторожное:
    где результат мог быть неточным (переполнение, вход вне области
    вызываемого эксперта), значение обнуляется, а ok ложен — такие строки
    досчитывает скалярный вычислитель."""
    N = X.shape[0]  # N — число входных строк
    op = expr.op  # имя операции
    if op == "var":  # переменная
        return X[:, int(expr.value)].copy(), np.ones(N, dtype=bool)  # столбец X[:, индекс] и все строки верны
    if op == "const":  # константа
        v = int(expr.value)  # значение константы
        if abs(v) > _VEC_MAG_LIMIT:  # модуль больше предела
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)  # нули и все строки неверны
        return np.full(N, v, dtype=np.int64), np.ones(N, dtype=bool)  # вектор из N копий значения и все строки верны
    if op == "call":  # вызов эксперта
        if env is None or expr.value not in env:  # среды нет или эксперта в ней нет
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)  # вернуть стоимость: нули и все строки неверны
        callee = env[expr.value]  # вызываемый эксперт
        arg_vals = []  # значения аргументов по столбцам
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        for a in expr.args:  # по аргументам
            v, aok = eval_expr_vec(a, X, env)  # значения и флаги аргумента
            arg_vals.append(v)  # добавить столбец значений
            ok &= aok  # ok = ok AND флаги аргумента
        if not np.any(ok):  # верных строк нет
            return np.zeros(N, dtype=np.int64), ok  # нули и все флаги ложны
        subX = np.column_stack(arg_vals)  # матрица аргументов (N, число аргументов) — столбцы значений
        for i, dom in enumerate(callee.domains):  # по аргументам вызываемого эксперта и их областям
            lo, hi = dom[0], dom[-1]  # lo = первый элемент области, hi = последний
            if len(dom) == hi - lo + 1 and dom == tuple(range(lo, hi + 1)):  # область — непрерывный отрезок lo..hi
                ok &= (subX[:, i] >= lo) & (subX[:, i] <= hi)  # итог: ok &= lo ≤ x_i ≤ hi
            else:  # иначе
                ok &= np.isin(subX[:, i], dom)  # ok &= x_i входит в область (точная проверка принадлежности)
        if callee.valid_rule == "nonzero_y":  # правило nonzero_y у вызываемого эксперта
            ok &= subX[:, 1] != 0  # итог: ok &= x_2 ≠ 0
        elif callee.valid_rule == "ordered_yz":  # правило ordered_yz
            ok &= subX[:, 1] <= subX[:, 2]  # итог: ok &= x_2 ≤ x_3
        if not np.any(ok):  # верных строк не осталось
            return np.zeros(N, dtype=np.int64), ok  # нули и флаги ok
        if callee.program.venn_mask is not None:  # вызываемая программа — диаграмма Венна
            idx = (np.clip(subX[:, 0], 0, 1) << 1) | np.clip(subX[:, 1], 0, 1)  # итог: idx = 2·clip(x_1, 0, 1) + clip(x_2, 0, 1)
            return ((int(callee.program.venn_mask) >> idx) & 1).astype(np.int64), ok  # значение = (маска >> idx) & 1
        res_all = np.zeros(N, dtype=np.int64)  # результат по всем строкам, изначально нули
        vals, ok2 = eval_expr_vec(callee.program.expr, subX[ok], env)  # вычислить выражение вызываемого эксперта на верных строках
        res_all[ok] = vals  # записать значения в верные строки
        ok_idx = np.flatnonzero(ok)  # номера верных строк
        ok[ok_idx[~ok2]] = False  # сбросить ok там, где вызов вернул флаг лжи
        return res_all, ok  # значения и флаги
    if op == "if_cond":  # условное выражение
        cv, cok = eval_expr_vec(expr.args[0], X, env)  # значения и флаги условия
        tv, tok = eval_expr_vec(expr.args[1], X, env)  # значения и флаги ветви «то»
        fv, fok = eval_expr_vec(expr.args[2], X, env)  # значения и флаги ветви «иначе»
        return np.where(cv != 0, tv, fv), cok & tok & fok  # условие ≠ 0 → ветвь «то», иначе «иначе»; ok = ok_условия AND ok_то AND ok_иначе
    if op == "poly":  # многочлен одного аргумента
        v, ok = eval_expr_vec(expr.args[0], X, env)  # значения и флаги аргумента x
        acc = np.zeros(N, dtype=np.int64)  # аккумулятор схемы Горнера, изначально нули
        for ci in reversed(list(expr.value or ())):  # по коэффициентам от старшего к младшему
            ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # строки с |acc| > 2^28 признать неверными
            acc = np.where(ok, acc * v + int(ci), 0)  # acc = acc·x + c_i в верных строках, иначе 0
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # окончательная проверка |acc| ≤ 2^28
        return acc, ok  # значения и флаги
    if op == "poly_nd":  # многочлен многих переменных
        acc = np.zeros(N, dtype=np.int64)  # сумма членов, изначально нули
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        for coef, powers in expr.value or ():  # по членам (коэффициент, показатели степени)
            term = np.full(N, int(coef), dtype=np.int64)  # член — вектор из N копий коэффициента
            for idx, p in enumerate(powers):  # по номерам переменных idx и показателям p
                if p:  # показатель не ноль
                    ok &= (np.abs(X[:, idx]) <= 1000) & (int(p) <= 6)  # ok &= |x_idx| ≤ 1000 и p ≤ 6
                    safe_x = np.where(ok, X[:, idx], 0)  # x_idx в верных строках, иначе 0
                    term = np.where(ok, term * (safe_x ** int(p)), 0)  # term = term·x_idx^p в верных строках, иначе 0
            acc = np.where(ok, acc + term, 0)  # acc = acc + term в верных строках, иначе 0
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # итог: ok &= |acc| ≤ 2^28
        return acc, ok  # значения и флаги
    if op == "anf":  # алгебраическая нормальная форма
        n_vars, coeffs = expr.value  # число переменных и список мономов с коэффициентами
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        mask_x = np.zeros(N, dtype=np.int64)  # битовая маска входа, изначально нули
        for i in range(n_vars):  # по переменным i = 0..n−1
            ok &= (X[:, i] == 0) | (X[:, i] == 1)  # итог: ok &= x_i ∈ {0, 1}
            mask_x |= np.where(X[:, i] != 0, 1 << (n_vars - 1 - i), 0)  # маска |= 2^(n−1−i) в строках, где x_i ≠ 0
        acc = np.zeros(N, dtype=np.int64)  # аккумулятор результата, изначально нули
        for s, c in coeffs:  # по мономам s и коэффициентам c
            acc ^= np.where((mask_x & s) == s, c, 0)  # acc = acc XOR c в строках, где s AND маска = s
        return acc, ok  # значения и флаги

    vs = []  # список столбцов значений аргументов
    ok = np.ones(N, dtype=bool)  # все строки пока верны
    for a in expr.args:  # по аргументам
        v, aok = eval_expr_vec(a, X, env)  # значения и флаги аргумента
        vs.append(v)  # добавить столбец
        ok &= aok  # ok = ok AND флаги аргумента
    if op == "neg":  # итог: neg
        res = -vs[0]  # итог: res = −x
    elif op == "abs":  # итог: abs
        res = np.abs(vs[0])  # итог: res = |x|
    elif op == "sign":  # итог: sign
        res = np.sign(vs[0]).astype(np.int64)  # res = знак x как int64
    elif op == "bit_not8":  # итог: bit_not8
        res = (~vs[0]) & 0xFF  # res = НЕ x AND 0xFF — инверсия младших 8 бит
    elif op == "clamp3":  # итог: clamp3
        res = np.clip(vs[0], -3, 3)  # итог: res = clip(x, −3, 3)
    elif op == "square":  # итог: square
        ok &= np.abs(vs[0]) <= 32768  # итог: ok &= |x| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x в верных строках, иначе 0
        res = s0 * s0  # итог: res = x²
    elif op == "cube":  # итог: cube
        ok &= np.abs(vs[0]) <= 1024  # итог: ok &= |x| ≤ 2^10
        s0 = np.where(ok, vs[0], 0)  # x в верных строках, иначе 0
        res = s0 * s0 * s0  # итог: res = x³
    elif op == "absdiff":  # итог: absdiff
        res = np.abs(vs[0] - vs[1])  # итог: res = |x_0 − x_1|
    elif op == "fma":  # итог: fma
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)  # ok &= |x_0| ≤ 2^15 и |x_1| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        res = s0 * s1 + vs[2]  # итог: res = x_0·x_1 + x_2
    elif op == "clamp":  # итог: clamp
        res = np.minimum(np.maximum(vs[0], vs[1]), vs[2])  # итог: res = min(max(x_0, x_1), x_2)
    elif op == "median3":  # итог: median3
        res = np.sort(np.stack(vs, axis=0), axis=0)[1]  # res = средний элемент построчной сортировки трёх аргументов
    elif op == "in_set":  # итог: in_set
        table = np.zeros(1 << 12, dtype=bool)  # таблица принадлежности на 4096 значений, изначально ложь
        for v in (expr.value or ()):  # по элементам множества
            if 0 <= int(v) < table.size:  # элемент попадает в таблицу: 0 ≤ v < 4096
                table[int(v)] = True  # пометить значение v как входящее
        res = table[np.clip(vs[0], 0, table.size - 1)]  # res = таблица[clip(x, 0, 4095)]
    elif op == "in_ranges":  # итог: in_ranges
        res = np.zeros_like(vs[0], dtype=bool)  # накопитель принадлежности, изначально ложь
        for lo, hi in (expr.value or ()):  # по диапазонам (lo, hi)
            res |= (vs[0] >= int(lo)) & (vs[0] <= int(hi))  # итог: res |= lo ≤ x ≤ hi
        res = res.astype(vs[0].dtype)  # привести к типу аргумента
    elif op == "mul3":  # итог: mul3
        ok &= (  # итог: ok &= (
            (np.abs(vs[0]) <= 1024) & (np.abs(vs[1]) <= 1024) & (np.abs(vs[2]) <= 1024)  # |x_0| ≤ 2^10 и |x_1| ≤ 2^10 и |x_2| ≤ 2^10
        )
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        s2 = np.where(ok, vs[2], 0)  # x_2 в верных строках, иначе 0
        res = s0 * s1 * s2  # mul3: итог res = x_0·x_1·x_2
    elif op == "add":  # итог: add
        res = vs[0] + vs[1]  # итог: res = x_0 + x_1
    elif op == "sub":  # итог: sub
        res = vs[0] - vs[1]  # итог: res = x_0 − x_1
    elif op == "mul":  # итог: mul
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)  # проверка: ok &= |x_0| ≤ 2^15 и |x_1| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        res = s0 * s1  # итог: res = x_0·x_1
    elif op == "min":  # итог: min
        res = np.minimum(vs[0], vs[1])  # итог: res = min(x_0, x_1)
    elif op == "max":  # итог: max
        res = np.maximum(vs[0], vs[1])  # итог: res = max(x_0, x_1)
    elif op == "eq":  # итог: eq
        res = (vs[0] == vs[1]).astype(np.int64)  # res = 1, если x_0 = x_1, иначе 0
    elif op == "ne":  # итог: ne
        res = (vs[0] != vs[1]).astype(np.int64)  # res = 1, если x_0 ≠ x_1, иначе 0
    elif op == "lt":  # итог: lt
        res = (vs[0] < vs[1]).astype(np.int64)  # res = 1, если x_0 < x_1, иначе 0
    elif op == "le":  # итог: le
        res = (vs[0] <= vs[1]).astype(np.int64)  # res = 1, если x_0 ≤ x_1, иначе 0
    elif op == "gt":  # итог: gt
        res = (vs[0] > vs[1]).astype(np.int64)  # res = 1, если x_0 > x_1, иначе 0
    elif op == "ge":  # итог: ge
        res = (vs[0] >= vs[1]).astype(np.int64)  # res = 1, если x_0 ≥ x_1, иначе 0
    elif op == "bit_and":  # итог: bit_and
        res = vs[0] & vs[1]  # итог: res = x_0 AND x_1
    elif op == "bit_or":  # итог: bit_or
        res = vs[0] | vs[1]  # итог: res = x_0 OR x_1
    elif op == "bit_xor":  # итог: bit_xor
        res = vs[0] ^ vs[1]  # итог: res = x_0 XOR x_1
    elif op == "bit_nand8":  # итог: bit_nand8
        res = (~(vs[0] & vs[1])) & 0xFF  # res = НЕ(x_0 AND x_1) по младшим 8 битам
    elif op == "bit_nor8":  # итог: bit_nor8
        res = (~(vs[0] | vs[1])) & 0xFF  # res = НЕ(x_0 OR x_1) по младшим 8 битам
    elif op == "bit_xnor8":  # итог: bit_xnor8
        res = (~(vs[0] ^ vs[1])) & 0xFF  # res = НЕ(x_0 XOR x_1) по младшим 8 битам
    elif op == "floordiv":  # итог: floordiv
        ok &= vs[1] != 0  # делитель 0 делает строку неверной: ok &= x_1 ≠ 0
        safe = np.where(vs[1] != 0, vs[1], 1)  # безопасный делитель: x_1, а где он равен 0 — 1
        res = np.floor_divide(vs[0], safe)  # res = ⌊x_0 / делитель⌋
    elif op == "mod":  # итог: mod
        ok &= vs[1] != 0  # итог: ok &= x_1 ≠ 0
        safe = np.where(vs[1] != 0, vs[1], 1)  # безопасный делитель: x_1, а где он равен 0 — 1
        res = np.mod(vs[0], safe)  # res = x_0 mod делитель
    elif op == "gcd":  # итог: gcd
        res = np.gcd(vs[0], vs[1])  # res = НОД(x_0, x_1)
    elif op == "lcm":  # итог: lcm
        res = np.lcm(vs[0], vs[1])  # res = НОК(x_0, x_1)
    elif op == "pow_int":  # итог: pow_int
        ok &= (vs[1] >= 0) & (vs[1] <= 8) & (np.abs(vs[0]) <= 32)  # ok &= 0 ≤ x_1 ≤ 8 и |x_0| ≤ 32
        sa = np.where(ok, vs[0], 0)  # основание в верных строках, иначе 0
        sb = np.where(ok, vs[1], 0)  # показатель в верных строках, иначе 0
        res = sa**sb  # итог: res = x_0^x_1
    else:  # неизвестная операция
        raise ValueError(f"unsupported vec op: {op}")  # ошибка: операция не поддерживается векторизованно
    ok &= np.abs(res) <= _VEC_MAG_LIMIT  # ok &= |res| ≤ 2^28 — результат не выходит за предел
    return res, ok  # вернуть значения и флаги


def candidate_programs(arity: int) -> list[Program]:
    """Начальные программы-кандидаты заданной арности: по одному простому выражению на каждую базовую операцию.

    Кандидаты — выражения только из переменных и одной операции. Дубликаты
    по тексту убираются. Возвращает список Program, отсортированный по
    (стоимость, читаемая запись)."""
    v = [Expr.var(i) for i in range(arity)]  # узлы переменных x_0, …, x_{arity−1}
    cands: list[Expr] = []  # список выражений-кандидатов
    if arity == 1:  # арность 1
        x = v[0]  # аргумент x
        cands += [  # добавить семь унарных выражений
            Expr("neg", (x,)),  # итог: neg(x)
            Expr("abs", (x,)),  # итог: abs(x)
            Expr("sign", (x,)),  # итог: sign(x)
            Expr("bit_not8", (x,)),  # итог: bit_not8(x)
            Expr("square", (x,)),  # итог: square(x)
            Expr("cube", (x,)),  # итог: cube(x)
            Expr("clamp3", (x,)),  # итог: clamp3(x)
        ]
    elif arity == 2:  # арность 2
        x, y = v  # аргументы x, y
        for op in BINARY_OPS:  # по всем бинарным операциям
            cands.append(Expr(op, (x, y)))  # кандидат op(x, y)
        cands.extend([Expr("sub", (y, x)), Expr("absdiff", (x, y))])  # плюс обратное вычитание y − x и |x − y|
    elif arity == 3:  # арность 3
        x, y, z = v  # аргументы x, y, z
        cands += [  # добавить четыре трёхместных выражения
            Expr("fma", (x, y, z)),  # итог: fma(x, y, z)
            Expr("clamp", (x, y, z)),  # итог: clamp(x, y, z)
            Expr("median3", (x, y, z)),  # итог: median3(x, y, z)
            Expr("mul3", (x, y, z)),  # итог: mul3(x, y, z)
        ]
    by_text = {}  # словарь «текст выражения -> выражение» для удаления дубликатов
    for e in cands:  # по кандидатам
        by_text.setdefault(e.pretty(), e)  # запомнить первого с таким текстом
    return [  # вернуть список программ
        Program(arity, e)  # программа заданной арности с выражением e
        for e in sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))  # в порядке (стоимость, текст)
    ]

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'REIFIED_OPS', 'Expr', 'Program', 'eval_expr_vec', 'candidate_programs',  # имя интерфейса
]  # конец перечня
