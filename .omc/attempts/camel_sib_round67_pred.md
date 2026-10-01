# Round 67 (prereg): camelmath-сиб — первый DATA-axis ход после R66

## Основание
R66: diversity сибов мертва (residual=шум); рост = новые данные.
D-плоскость: camelmath w=0.10 поднимает D с 0.662 до 0.690 (+4%);
openhermes D понижает (отклонён по протоколу D(w)).

## Слепой прогноз
1. Solo gate CE (канон-окна): 3.56–3.60 (сравнимо с seed-сибами;
   новый корпус 10% не ломает общий микс).
2. Per-corpus: camelmath-домен недообучен vs openwebmath — но это
   NEW информация; R66-тест: CE(residual solo vs seed-sib) — если
   residual сиба теперь информативен (CE << ln V), data-diversity
   РЕАЛЬНАЯ → полный 3-сиб цикл оправдан.
3. Go-критерий: residual-тест (R66) на паре camel-сиб × seed-сиб.

## Стоп
Solo > 3.70 → вес 0.10 токсичен, пересборка микса.
