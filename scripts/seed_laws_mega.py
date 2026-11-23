"""WAVE 240: MEGA law seeding — ONE .chat file, every example
exactly once (dedup by content hash).

Sections:
  A. Alphabets (latin/cyrillic/greek/digits/hex/base64/ascii85)
  B. Encodings (UTF-8 byte anatomy, UTF-16 BOM, base64, hex,
     ROT13, Caesar shifts, URL-escape)
  C. Bit hacks (graphics.stanford.edu/~seander/bithacks.html):
     swap-xor, popcount, round-up-pow2, abs, min/max, bit-reversal,
     power-of-2 check, mask tricks, gray code, parity
Each example appears ONCE globally.
"""
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

INBOX = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "checkpoints", "inbox")

_seen = set()


def add(lines: list[str], s: str) -> None:
    """Append line if its content hash was not seen."""
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    if h in _seen:
        return
    _seen.add(h)
    lines.append(s)


def section_alphabets(lines):
    add(lines, "## АЛФАВИТЫ: следующая буква = предыдущая + 1")
    # latin lower
    add(lines, "abcdefghijklmnopqrstuvwxyz -> each next = prev+1")
    # latin upper
    add(lines, "ABCDEFGHIJKLMNOPQRSTUVWXYZ -> each next = prev+1")
    # cyrillic
    add(lines, "абвгдежзийклмнопрстуфхцчшщъыьэюя -> each next = prev+1")
    # greek
    add(lines, "αβγδεζηθικλμνξοπρστυφχψω -> each next = prev+1")
    # digits
    add(lines, "0123456789 -> each next = prev+1")
    # hex digits
    add(lines, "0123456789abcdef -> hex alphabet order")
    # base64 alphabet
    add(lines, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
               "0123456789+/ -> base64 alphabet order")
    # reversed alphabets
    add(lines, "zyxwvutsrqponmlkjihgfedcba -> each next = prev-1")
    add(lines, "9876543210 -> each next = prev-1")


def section_encodings(lines):
    add(lines, "## КОДИРОВКИ: байтовая анатомия UTF-8")
    # UTF-8 anatomy of ASCII (1 byte)
    add(lines, "UTF-8: ASCII 'A'=0x41 (1 байт), 'Кирилл 'а'=0xD0B0 "
               "(2 байта), '€'=0xE282AC (3 байта)")
    # byte-length pattern
    for ch, enc in (("A", "41"), ("а", "D0B0"), ("Я", "D0AF"),
                    ("€", "E282AC"), ("中", "E4B8AD")):
        add(lines, f"UTF-8 байты '{ch}' = {enc}")
    # ROT13 on alphabet
    add(lines, "ROT13: abcdefghijklmnopqrstuvwxyz -> "
               "nopqrstuvwxyzabcdefghijklm (сдвиг 13)")
    # Caesar shifts on a sample word
    for shift in (1, 3, 5):
        w = "hello"
        enc = "".join(chr((ord(c) - 97 + shift) % 26 + 97) for c in w)
        add(lines, f"Цезарь +{shift}: hello -> {enc} (каждая буква "
                   f"сдвигается на {shift})")
    # base64 of known strings (each unique)
    import base64
    for s in (b"Man", b"Hello", b"a", b"ab", b"abc", b"test"):
        add(lines, f"base64({s.decode()}) = "
                   f"{base64.b64encode(s).decode()}")
    # hex encoding
    for s in (b"abc", b"test", b"ABC"):
        add(lines, f"hex({s.decode()}) = {s.hex()}")
    # URL escape
    for s in ("a b", "a&b", "a=b"):
        esc = (s.replace(" ", "%20").replace("&", "%26")
               .replace("=", "%3D"))
        add(lines, f"URL-escape '{s}' -> {esc}")


def section_bithacks(lines):
    add(lines, "## БИТОВЫЕ ТРЮКИ (graphics.stanford.edu/~seander/"
               "bithacks.html)")
    # XOR swap
    add(lines, "XOR-swap: a^=b; b^=a; a^=b — обмен без временной "
               "переменной. Пример: a=5,b=9 -> a=9,b=5")
    # popcount (Brian Kernighan)
    for v in (0b1011, 0b1111, 0b100000, 0b11111111, 7, 255, 128):
        add(lines, f"popcount({v}) = {bin(v).count('1')} "
                   f"(число единичных бит в {bin(v)})")
    # round up to power of 2
    for v in (5, 17, 100, 255, 256, 1000):
        r = 1
        while r < v:
            r <<= 1
        add(lines, f"round_up_pow2({v}) = {r} (ближайшая степень "
                   f"двойки сверху)")
    # power of 2 check
    for v, is_p in ((1, True), (2, True), (3, False), (64, True),
                    (100, False), (128, True)):
        add(lines, f"is_power_of_2({v}) = {is_p and v != 0} "
                   f"(v & (v-1)) == 0 и v > 0")
    # abs without branch
    for v in (-5, 7, -128, 42):
        add(lines, f"abs({v}) = {abs(v)} (маска m = x>>31; "
                   f"(x+m)^m)")
    # min/max without branch
    add(lines, "min(x,y) = y ^ ((x ^ y) & -(x < y)); "
               "max(x,y) = x ^ ((x ^ y) & -(x < y))")
    add(lines, "пример: min(3,7)=3, max(3,7)=7")
    # bit reversal (8-bit)
    for v in (0b1, 0b110, 0b1010, 0b11110000):
        rev = int(format(v, '08b')[::-1], 2)
        add(lines, f"bit_reverse_8({v:3}) = {rev:3} "
                   f"({format(v, '08b')} -> {format(rev, '08b')})")
    # parity
    for v in (0b1011, 0b1111, 0b1100, 7):
        p = bin(v).count('1') % 2
        add(lines, f"parity({bin(v)}) = {p} (XOR всех бит: "
                   f"{bin(v).count('1')} единиц -> {'нечёт' if p else 'чёт'})")
    # gray code
    for v in (0, 1, 2, 3, 4, 5, 7, 10):
        g = v ^ (v >> 1)
        add(lines, f"gray_code({v}) = {g} (v ^ (v>>1))")
    # masks
    add(lines, "младший бит: x & 1; сброс младшего единичного: "
               "x & (x-1); изоляция младшего единичного: x & -x")
    add(lines, "пример: x=0b1010 -> x&(x-1)=0b1000, x&-x=0b0010")
    # swap nibbles
    for v in (0xAB, 0x12, 0xF0):
        sw = ((v & 0x0F) << 4) | ((v >> 4) & 0x0F)
        add(lines, f"swap_nibbles({v:#04x}) = {sw:#04x}")
    # sign extension tricks
    add(lines, "знак числа: x >> 31 даёт -1 для отрицательных, "
               "0 для неотрицательных (32-бит int)")
    # conditional set without branch
    add(lines, "условная установка без ветвления: r = y ^ ((x ^ y) "
               "& -(cond)) — выбирает x если cond, иначе y")


def main() -> None:
    os.makedirs(INBOX, exist_ok=True)
    lines: list[str] = []
    section_alphabets(lines)
    section_encodings(lines)
    section_bithacks(lines)
    # заголовок
    header = ("УЧЕБНЫЙ ПАКЕТ ЗАКОНОВ (wave240): каждый пример ниже "
              "встречается ровно один раз. Выучи все закономерности.\n\n")
    content = header + "\n".join(lines)
    path = os.path.join(INBOX, "mega_laws.chat")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"queued: {path}")
    print(f"строк: {len(lines)} | всего символов: {len(content)} | "
          f"уникальных (по хэшу): {len(_seen)}")


if __name__ == "__main__":
    main()
