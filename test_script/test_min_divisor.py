def make_divisible(v,divisor=8,min_value:int=None) -> int:
    if min_value is None:
        min_value = 8
    new_v = max(min_value,int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# test
if __name__ == "__main__":
    inplanes = 16
    scale = 0.5
    c_list = [
        16,24,40,80,112,160
    ]
    for c in c_list:
        cc = make_divisible(c * scale)
        print("the input value is {},output value is {}".format(
            c * scale,
            cc
        ))