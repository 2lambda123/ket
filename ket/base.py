from __future__ import annotations
from math import pi
import json
from .clib.libket import *
from .clib.wrapper import from_list_to_c_vector, from_u8_to_str

__all__ = ['quant', 'future', 'dump', 'qc_int',
           'exec_quantum', 'quantum_metrics', 'quantum_code', 'quantum_exec_time', 'quantum_exec_timeout']


class quant:
    r"""Create list of qubits

    Allocate ``size`` qubits in the state :math:`\left|0\right>`.

    If ``dirty`` is ``True``, allocate ``size`` qubits in an unknown state.

    warning:
        Using dirty qubits may have side effects due to previous entanglements.

    Qubits allocated using the ``with`` statement must be free at the end of the scope.

    Example:

    .. code-block:: ket

        a = H(quant()) 
        b = X(quant())
        with quant() as aux: 
            with around(H, aux):
                with control(aux):
                    swap(a, b)
            result = measure(aux)
            if result == 1:
                X(aux) 
            aux.free() 

    :Qubit Indexing:

    Use brackets to index qubits as in a ``list`` and use ``+`` to concatenate
    two :class:`~ket.libket.quant`.

    Example:

    .. code-block:: ket

        q = quant(20)        
        head, tail = q[0], q[1:]
        init, last = q[:-1], q[-1]
        even = q[::2]
        odd = q[1::2]
        reverse = reversed(q) # invert qubits order

        a, b = quant(2) # |a⟩, |b⟩
        c = a+b         # |c⟩ = |a⟩|b⟩ 

    Args:
        size: The number of qubits to allocate.
        dirty: If ``True``, allocate ``size`` qubits at an unknown state.
        qubits: Initialize the qubit list without allocating. Intended for internal use.
    """

    def __init__(self, size: int = 1, dirty: bool = False, *, qubits: list[qubit] | None = None):
        if qubits is not None:
            self.qubits = qubits
        else:
            self.qubits = [qubit(process_top().allocate_qubit(dirty))
                           for _ in range(size)]

    def __add__(self, other: quant) -> quant:
        return quant(qubits=self.qubits+other.qubits)

    def at(self, index: list[int]) -> quant:
        r"""Return qubits at ``index``

        Create a new :class:`~ket.libket.quant` with the qubit references at the
        position defined by the ``index`` list.

        :Example:

        .. code-block:: ket

            q = quant(20)        
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        Args:
            index: List of indexes.
        """

        return quant(qubits=[self.qubits[i] for i in index])

    def free(self, dirty: bool = False):
        r"""Free the qubits

        All qubits must be at the state :math:`\left|0\right>` before the call,
        otherwise set the ``dirty`` param to ``True``.

        Warning: 
            No check is applied to see if the qubits are at state
            :math:`\left|0\right>`.

        Args:
            dirty: Set ``True`` to free dirty qubits.
        """

        for qubit in self.qubits:
            process_top().free_qubit(qubit, dirty)

    def is_free(self) -> bool:
        """Return ``True`` when all qubits are free"""
        return all(not qubit.allocated().value for qubit in self.qubits)

    def __reversed__(self):
        return quant(qubits=reversed(self.qubits))

    def __getitem__(self, key):
        qubits = self.qubits.__getitem__(key)
        return quant(qubits=qubits if isinstance(qubits, list) else [qubits])

    class iter:
        def __init__(self, q):
            self.q = q
            self.idx = -1
            self.size = len(q.qubits)

        def __next__(self):
            self.idx += 1
            if self.idx < self.size:
                return self.q[self.idx]
            raise StopIteration

        def __iter__(self):
            return self

    def __iter__(self):
        return self.iter(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if not self.is_free():
            raise RuntimeError('non-free quant at the end of scope')

    def __len__(self):
        return len(self.qubits)

    def __repr__(self):
        return f"<Ket 'quant' {[(q.pid().value, q.index().value) for q in self.qubits]}>"


class future:
    """64-bits integer on the quantum computer

    Store a reference to a 64-bits integer available in the quantum computer. 

    The integer value are available to the classical computer only after the
    quantum execution.

    The following binary operations are available between
    :class:`~ket.libket.future` variables and ``int``: 

        ``==``, ``!=``, ``<``, ``<=``,
        ``>``, ``>=``, ``+``, ``-``, ``*``, ``/``, ``<<``, ``>>``, ``and``,
        ``xor``, and ``or``.

    A new :class:`~ket.libket.future` variable is created with a quantum
    :func:`~ket.standard.measure` (1) , binary operation with a
    :class:`~ket.libket.future` (2), or directly initialization with a ``int``
    (2).

    .. code-block:: ket

        q = H(quant(2))
        a = measure(q) # 1
        b = a*3        # 2
        c = qc_int(42) # 3



    Writing to the attribute ``value`` of a :class:`~ket.libket.future` variable
    passes the information to the quantum computer. 
    Reading the attribute ``value`` triggers the quantum execution. 

    If the test expression of an ``if-then-else`` or ``while`` is type future,
    Ket passes the statement to the quantum computer.

    :Example:

    .. code-block:: ket

        q = quant(2)
        with quant() as aux:
            # Create variable done on the quantum computer
            done = qc_int(False) 
            while done != True:
                H(q)
                ctrl(q, X, aux)
                res = measure(aux)
                if res == 0:
                    # Update variable done on the quantum computer
                    done.value = True
                else:
                    X(q+aux)
            aux.free()
        # Get the measurement from the quantum computer
        # triggering the quantum execution
        result = measure(q).value 

    """

    def __init__(self, value: int | libket_future):
        if isinstance(value, int):
            self.base = qc_int(value).base
        else:
            self.base = value
        self._value = None

    def __getattr__(self, name):
        if name == "value":
            if self._value is None:
                exec_quantum()
                self._value = self.base.value().value
            return self._value
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "value":
            if not isinstance(value, future):
                value = qc_int(value)
            process_top().int_set(self.base, value)
        else:
            super().__setattr__(name, value)

    @property
    def available(self) -> bool:
        return self.base.available().value

    @property
    def index(self) -> int:
        return self.base.index().value

    @property
    def pid(self) -> int:
        return self.base.pid().value

    def __add__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(ADD, self.base, other.base)))

    def __sub__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SUB, self.base, other.base)))

    def __mul__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(MUL, self.base, other.base)))

    def __truediv__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(DIV, self.base, other.base)))

    def __floordiv__(self, other: future | int) -> future:
        return self.__truediv__(other)

    def __lshift__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SLL, self.base, other.base)))

    def __rshift__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SRL, self.base, other.base)))

    def __and__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(AND, self.base, other.base)))

    def __xor__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(XOR, self.base, other.base)))

    def __or__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(OR, self.base, other.base)))

    def __radd__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(ADD, other.base, self.base)))

    def __rsub__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SUB, other.base, self.base)))

    def __rmul__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(MUL, other.base, self.base)))

    def __rtruediv__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(DIV, other.base, self.base)))

    def __rfloordiv__(self, other: future | int) -> future:
        return self.__rtruediv__(other)

    def __rlshift__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SLL, other.base, self.base)))

    def __rrshift__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(SRL, other.base, self.base)))

    def __rand__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(AND, other.base, self.base)))

    def __rxor__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(XOR, other.base, self.base)))

    def __ror__(self, other: future | int) -> future:
        other = qc_int(other)
        return future(libket_future(process_top().add_int_op(OR, other.base, self.base)))

    def __lt__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(LT, self.base, other.base)))

    def __le__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(LEQ, self.base, other.base)))

    def __eq__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(EQ, self.base, other.base)))

    def __ne__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(NEQ, self.base, other.base)))

    def __gt__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(GT, self.base, other.base)))

    def __ge__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return future(libket_future(process_top().add_int_op(GEQ, self.base, other.base)))

    def __repr__(self):
        return f"<Ket 'future' {self.pid, self.index}>"


class dump:
    """Create a snapshot with the current quantum state of ``qubits``.

    Gathering any information from a :class:`~ket.libket.dump` triggers the quantum execution.

    :Example:

    .. code-block:: ket

        a, b = quant(2)
        with around(cnot(H, I), a, b):
            Y(a)
            inside = dump(a+b)
        outside = dump(a+b)

        print('inside:')
        print(inside.show())
        #inside:
        #|01⟩    (50.00%)
        #         -0.707107i     ≅     -i/√2
        #|10⟩    (50.00%)
        #          0.707107i     ≅      i/√2
        print('outside:')
        print(outside.show())
        #outside:
        #|11⟩    (100.00%)
        #         -1.000000i     ≅     -i/√1

    :param qubits: Qubits to dump.
    """

    def __init__(self, qubits: quant):
        self.base = libket_dump(process_top().dump(
            *from_list_to_c_vector(qubits.qubits)))
        self.size = len(qubits.qubits)

    @property
    def states(self) -> list[int]:
        """List of basis states"""

        if not self.base.available().value:
            exec_quantum()

        size = self.base.states_size().value

        for i in range(size):
            state, state_size = self.base.state(i)
            yield int(''.join(f'{state[j]:064b}' for j in range(state_size.value)), 2)

    @property
    def amplitudes(self) -> list[complex]:
        """List of probability amplitudes"""

        if not self.available:
            exec_quantum()

        real, size = self.base.amplitudes_real()
        imag, _size = self.base.amplitudes_imag()
        assert(size.value == _size.value)

        for i in range(size.value):
            yield real[i]+imag[i]*1j

    @property
    def probability(self) -> list[float]:
        """List of measurement probability"""

        for amp in self.amplitudes:
            yield abs(amp)**2

    def show(self, format: str | None = None) -> str:
        r"""Return the quantum state as a string

        Use the format starting to change the print format of the basis states:

        * ``i``: print the state in the decimal base
        * ``b``: print the state in the binary base (default)
        * ``i|b<n>``: separate the ``n`` first qubits, the remaining print in the binary base
        * ``i|b<n1>:i|b<n2>[:i|b<n3>...]``: separate the ``n1, n2, n3, ...`` first qubits

        :Example:

        .. code-block:: ket

            q = quant(19)
            X(ctrl(H(q[0]), X, q[1:])[1::2])
            d = dump(q)

            print(d.show('i'))
            #|87381⟩ (50.00%)
            # 0.707107               ≅      1/√2
            #|436906⟩        (50.00%)
            # 0.707107               ≅      1/√2
            print(d.show('b'))
            #|0010101010101010101⟩   (50.00%)
            # 0.707107               ≅      1/√2
            #|1101010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2
            print(d.show('i4'))
            #|2⟩|101010101010101⟩    (50.00%)
            # 0.707107               ≅      1/√2
            #|13⟩|010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2
            print(d.show('b5:i4'))
            #|00101⟩|5⟩|0101010101⟩  (50.00%)
            # 0.707107               ≅      1/√2
            #|11010⟩|10⟩|1010101010⟩ (50.00%)
            # 0.707107               ≅      1/√2

        Args:
            format: Format string that matches ``(i|b)\d*(:(i|b)\d+)*``.
        """

        if format is not None:
            if format == 'b' or format == 'i':
                format += str(self.size)
            fmt = []
            count = 0
            for b, size in map(lambda f: (f[0], int(f[1:])), format.split(':')):
                fmt.append((b, count, count+size))
                count += size
            if count < self.size:
                fmt.append(('b', count, self.size))
        else:
            fmt = [('b', 0, self.size)]

        def fmt_ket(state, begin, end, f):
            return f'|{state[begin:end]}⟩' if f == 'b' else f'|{int(state[begin:end], base=2)}⟩'

        def state_amp_str(state, amp):
            dump_str = ''.join(
                fmt_ket(f'{state:0{self.size}b}', b, e, f) for f, b, e in fmt)
            dump_str += f"\t({100*abs(amp)**2:.2f}%)\n"
            real = abs(amp.real) > 1e-10
            real_l0 = amp.real < 0

            imag = abs(amp.imag) > 1e-10
            imag_l0 = amp.imag < 0

            sqrt_dem = 1/abs(amp)**2
            use_sqrt = abs(round(sqrt_dem)-sqrt_dem) < .001
            sqrt_dem = f'/√{round(1/abs(amp)**2)}'

            if real and imag:
                sqrt_dem = f'/√{round(2*(1/abs(amp)**2))}'
                sqrt_num = ('(-1' if real_l0 else ' (1') + \
                    ('-i' if imag_l0 else '+i')
                sqrt_str = f'\t≅ {sqrt_num}){sqrt_dem}' if use_sqrt and (
                    abs(amp.real)-abs(amp.real) < 1e-10) else ''
                dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i"+sqrt_str
            elif real:
                sqrt_num = '  -1' if real_l0 else '   1'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}' if use_sqrt else ''
                dump_str += f"{amp.real:9.6f}       "+sqrt_str
            else:
                sqrt_num = '  -i' if imag_l0 else '   i'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}' if use_sqrt else ''
                dump_str += f" {amp.imag:17.6f}i"+sqrt_str

            return dump_str

        return '\n'.join(state_amp_str(state, amp) for state, amp in sorted(zip(self.states, self.amplitudes), key=lambda k: k[0]))

    @property
    def expected_values(self):
        """X, Y, and Z expected values for one qubit"""

        if self.size != 1:
            raise RuntimeError(
                'Cannot calculate X, Y, and Z expected values from a dump with more than 1 qubit')

        def exp_x(alpha, beta):
            return (beta.conjugate()*alpha+alpha.conjugate()*beta).real

        def exp_y(alpha, beta):
            return (1j*beta.conjugate() * alpha-1j*alpha.conjugate()*beta).real

        def exp_z(alpha, beta):
            return pow(abs(alpha), 2)-pow(abs(beta), 2)

        alpha = 0
        beta = 0
        for a, s in zip(self.amplitudes, self.states):
            if s == 0:
                alpha = a
            else:
                beta = a
        return [exp_x(alpha, beta), exp_y(alpha, beta), exp_z(alpha, beta)]

    def sphere(self):
        """Result a Bloch sphere

        QuTiP and Matplotlib are needed to generate and plot the sphere.
        """
        try:
            import qutip
        except ImportError as e:
            from sys import stderr
            print("Unable to import QuTiP, try installing:", file=stderr)
            print("\tpip install qutip", file=stderr)
            raise e

        b = qutip.Bloch()
        b.add_vectors(self.expected_values)
        return b

    @property
    def available(self) -> bool:
        return self.base.available().value

    @property
    def index(self) -> int:
        return self.base.index().value

    @property
    def pid(self) -> int:
        return self.base.pid().value

    def __repr__(self) -> str:
        return f"<Ket 'dump' {self.pid, self.index}>"


class label:

    def __init__(self):
        self.base = libket_label(process_top().get_label())

    @property
    def index(self) -> int:
        return self.base.index().value

    @property
    def pid(self) -> int:
        return self.base.pid().value

    def begin(self):
        process_top().open_block(self.base)

    def __repr__(self) -> str:
        return f"<Ket 'label' {self.pid, self.index}>"


_process_count = 1
_process_stack = [process(0)]


def process_begin():
    global _process_count
    global _process_stack
    _process_stack.append(process(_process_count))
    _process_count += 1


def process_end() -> process:
    global _process_stack
    return _process_stack.pop()


def process_top() -> process:
    global _process_stack
    return _process_stack[-1]


quantum_execution_target = None
_exec_time = None


def set_quantum_execution_target(func):
    global quantum_execution_target
    quantum_execution_target = func


def exec_quantum():
    """Call the quantum execution"""

    global quantum_execution_target
    global _exec_time

    process_top().prepare_for_execution()

    error = None
    try:
        quantum_execution_target(process_top())
        _exec_time = process_end().exec_time().value
    except Exception as e:
        error = e
    process_begin()
    if error:
        raise error


def qc_int(value: int) -> future:
    """Instantiate an integer on the quantum computer

    args:
        value: Initial value.
    """

    return future(libket_future(process_top().int_new(value)))


def base_measure(qubits: quant) -> future:
    if not len(qubits):
        return None

    size = len(qubits)
    if size <= 64:
        return future(libket_future(process_top().measure(*from_list_to_c_vector(qubits.qubits))))
    else:
        return [future(libket_future(process_top().measure(*from_list_to_c_vector(qubits.qubits[i:min(i+63, size)])))) for i in reversed(range(0, size, 63))]


def plugin(name: str, args: str, qubits: quant):
    """Apply plugin

    .. note::

        Plugin availability depends on the quantum execution target.

    args:
        name: Plugin name.
        args: Plugin argument string.
        qubits: Affected qubits.
    """
    process_top().apply_plugin(name.encode(), args.encode(),
                               *from_list_to_c_vector(qubits.qubits))


def ctrl_push(qubits: quant):
    return process_top().ctrl_push(*from_list_to_c_vector(qubits.qubits))


def ctrl_pop():
    return process_top().ctrl_pop()


def adj_begin():
    return process_top().adj_begin()


def adj_end():
    return process_top().adj_end()


def jump(goto: label):
    return process_top().jump(goto.base)


def branch(test: future, then: label, otherwise: label):
    return process_top().branch(test.base, then.base, otherwise.base)


def base_X(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PAULI_X, 0.0, qubit)


def base_Y(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PAULI_Y, 0.0, qubit)


def base_Z(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PAULI_Z, 0.0, qubit)


def base_H(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(HADAMARD, 0.0, qubit)


def base_S(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PHASE, pi/2, qubit)


def base_SD(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PHASE, -pi/2, qubit)


def base_T(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PHASE, pi/4, qubit)


def base_TD(q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PHASE, -pi/4, qubit)


def base_phase(lambda_, q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(PHASE, lambda_, qubit)


def base_RX(theta, q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(RX, theta, qubit)


def base_RY(theta, q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(RY, theta, qubit)


def base_RZ(theta, q: quant):
    for qubit in q.qubits:
        process_top().apply_gate(RZ, theta, qubit)


def quantum_metrics():
    process_top().serialize_metrics(JSON)
    return json.loads(from_u8_to_str(*process_top().get_serialized_metrics()[:-1]))


def quantum_code():
    process_top().serialize_quantum_code(JSON)
    return json.loads(from_u8_to_str(*process_top().get_serialized_quantum_code()[:-1]))


def quantum_exec_time():
    return _exec_time


def quantum_exec_timeout(timeout: int):
    return process_top().set_timeout(timeout)