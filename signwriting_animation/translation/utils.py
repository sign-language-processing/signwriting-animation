import copy
import itertools
from itertools import chain

from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.formats.sign_to_fsw import sign_to_fsw
from signwriting.tokenizer import normalize_signwriting
from signwriting.types import Sign


def factor_signwriting(fsw: str):
    signs = normalize_signwriting(fsw).split(" ")
    signs = [fsw_to_sign(sign) for sign in signs]
    for sign in signs:  # override box position same as the tokenizer does
        sign["box"]["position"] = (500, 500)
    units = list(chain.from_iterable([[sign["box"]] + sign["symbols"] for sign in signs]))

    return [
        [s["symbol"][:4] for s in units],
        ["c" + (s["symbol"][4] if len(s["symbol"]) > 4 else '0') for s in units],
        ["r" + (s["symbol"][5] if len(s["symbol"]) > 5 else '0') for s in units],
        ["p" + str(s["position"][0]) for s in units],
        ["p" + str(s["position"][1]) for s in units],
    ]


def permute_sign(sign: Sign, max_permutations=50):
    permutations = itertools.permutations(sign["symbols"])
    permutations = itertools.islice(permutations, max_permutations)
    for perm in permutations:
        new_sign = copy.deepcopy(sign)
        new_sign["symbols"] = list(perm)
        yield sign_to_fsw(new_sign)


def permute_signwriting(fsw: str, max_permutations=50):
    signs = normalize_signwriting(fsw).split(" ")
    signs = [fsw_to_sign(sign) for sign in signs]
    permutations = [list(permute_sign(sign)) for sign in signs]
    product = itertools.product(*permutations)
    product = itertools.islice(product, max_permutations)
    for perm in product:
        yield " ".join(perm)


def factored_signwriting_str(fsw: str):
    factors = factor_signwriting(fsw)
    return " ".join(["|".join(f) for f in zip(*factors)])


if __name__ == "__main__":
    print("\n".join(permute_signwriting(
        "AS20310S26b02S33100M521x547S33100482x483S20310506x500S26b02503x520")))
