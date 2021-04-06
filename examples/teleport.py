from ket import quant, x, z, h, measure, ctrl, code_ket
from ket.lib import bell

@code_ket
def teleport(alice : quant) -> quant:
    alice_b, bob_b = bell()

    ctrl(alice, x, alice_b)
    h(alice)

    m0 = measure(alice)
    m1 = measure(alice_b)

    if m1 == 1:
        x(bob_b)
    if m0 == 1:
        z(bob_b)

    return bob_b

alice = quant(1)         # alice = |0⟩
h(alice)                 # alice = |+⟩
z(alice)                 # alice = |–⟩
bob = teleport(alice)    # bob  <- alice
h(bob)                   # bob   = |1⟩
bob_m = measure(bob)

print('Expected measure 1, result =', bob_m.get())
# Expected measure 1, result = 1     