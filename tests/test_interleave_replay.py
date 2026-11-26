"""Повтор старых доменов чередуется с новым; хвост нового домена под контроль остаётся чистым."""
from hagi.selfdev.unified import _interleave_replay  # сборка обучающей выборки


def test_replay_is_interleaved_and_tail_is_pure():
    """Повтор разнесён по всей выборке, порядок байтов внутри каждого домена сохранён, хвост — только новый домен."""
    fresh = bytes(range(2, 256)) * 100  # новый домен: 25 400 байтов без байта 1
    replay = bytes([1]) * 30000  # повтор старого: только байт 1
    out = _interleave_replay(fresh, replay)  # сборка
    cut = int(0.8 * len(fresh))  # граница контрольного хвоста
    assert len(out) == len(fresh) + len(replay)  # ничего не потеряно и не добавлено
    assert out[-(len(fresh) - cut):] == fresh[cut:]  # хвост — чисто новый домен
    assert bytes(b for b in out if b != 1) == fresh and out.count(1) == len(replay)  # порядок каждого домена сохранён
    blocks = [i // 4096 for i, b in enumerate(out) if b == 1]  # номера блоков с повтором
    assert blocks[0] <= 2 and blocks[-1] >= len(out) // 4096 - 3  # повтор от начала и почти до конца выборки
    assert max(y - x for x, y in zip(blocks, blocks[1:])) <= 2  # без длинных прогонов без повтора


def test_short_inputs_are_simply_joined():
    """Без повтора или на коротком домене выборка — простое соединение."""
    assert _interleave_replay(b"ab", b"cd") == b"abcd"  # короткий новый домен
    assert _interleave_replay(b"x" * 50000, b"") == b"x" * 50000  # пустой повтор
