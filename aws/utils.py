import time
import math

from typing import Iterable

from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction
from braket.circuits.qubit import QubitInput
from braket.circuits.qubit_set import QubitSet

###############################################################################
#                                                                               
#                               Utility methods                                 
#                                                                               
###############################################################################

# Copied from https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m
def modInverse(a, modulus):
    initial_modulus = modulus
    t = 0
    s = 1

    while a > 1:
        quotient = a // modulus
        prev_modulus = modulus 

        modulus = a % modulus
        a = prev_modulus
        prev_modulus = t

        t = s - quotient * t 
        s = prev_modulus

    if s < 0:
        s = s + initial_modulus 

    return s 

def addToCircuit(circuit: Circuit, instruction: Instruction):
    if instruction is not None:
        circuit.add(instruction)

def invertInstruction(instruction: Instruction) -> Instruction:
    if instruction is None:
        return None

    operator = instruction.operator
    target = instruction.target
    if isinstance(operator, Gate.CPhaseShift):
        return Instruction(Gate.CPhaseShift(-operator.angle), target = target)
    if isinstance(operator, Gate.PhaseShift):
        return Instruction(Gate.PhaseShift(-operator.angle), target = target)
    elif isinstance(operator, Gate.CNot):
        return Instruction(Gate.CNot(), target = target)
    elif isinstance(operator, Gate.CCNot):
        return Instruction(Gate.CCNot(), target = target)
    elif isinstance(operator, Gate.X):
        return Instruction(Gate.X(), target = target)
    elif isinstance(operator, Gate.H):
        return Instruction(Gate.H(), target = target)
    elif isinstance(operator, Gate.Swap):
        return Instruction(Gate.Swap(), target = target)
    elif isinstance(operator, Gate.CSwap):
        return Instruction(Gate.CSwap(), target = target)
    else:
        raise ValueError("Invalid operator type within instruction passed to invertInstruction")

def invertInstructions(instructions: Iterable[Instruction]) -> Iterable[Instruction]:
    def _flatten(other):
        if isinstance(other, Iterable) and not isinstance(other, str):
            for item in other:
                yield from _flatten(item)
        else:
            yield other

    return [invertInstruction(instruction) for instruction in reversed(list(_flatten(instructions))) ]

def setCircuitBits(circuit: Circuit, bit_list: Iterable[int], offset: int) -> Circuit:
    for idx in range(len(bit_list)):
        if bit_list[idx] > 0:
            circuit.x(offset + idx)

    return circuit

def getNumberAsBits(number: int, bits_per_number: int = -1):
    bit_list = []

    if bits_per_number < 0:
        bits_per_number = math.ceil(math.log(number, 2))

    for idx in range(bits_per_number):
        bit_list.append(number % 2)
        number = math.floor(number / 2)

    return bit_list

def getNumberFromBitString(bit_string: str, num_digits: int, offset: int) -> int:
    bit_list = bit_string[offset:offset + num_digits]

    number = 0
    current_power = 1
    for idx in range(len(bit_list)):
        if bit_list[idx] == '1':
            number += current_power

        current_power *= 2

    return number

###############################################################################
#                                                                               
#                                Gates & Circuits                               
#                                                                               
###############################################################################
"""
Method `QFT` represents a quantum Fourier transform sub-circuit.

Args:
    target (QubitInput): Stores the resulting quantum Fourier transform value.
"""
def QFT(target: QubitInput) -> Circuit:
    circuit = Circuit()

    num_qubits = len(target)
    two_pi_radians = 2 * math.pi
    for target_idx in range(num_qubits - 1, -1, -1):
        addToCircuit(circuit, Gate.H.h(target[target_idx]))

        for control_idx in range(target_idx - 1, -1, -1):
            addToCircuit(circuit, Gate.CPhaseShift.cphaseshift(
                    target[control_idx],
                    target[target_idx],
                    two_pi_radians / (2 ** (target_idx - control_idx + 1))
            ))

    return circuit

def buildTestQFTCircuit(number_int, bits_per_number):
    number_bits = getNumberAsBits(number_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    number = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, number_bits, bit_index)
    bit_index += bits_per_number

    circuit.add(QFT(number))

    return circuit, number

bits_per_number = 4

if __name__ == "__main__":
    def testAllNumericQFT(device):
        for number in range(2 ** bits_per_number):
            test_circuit, number = buildTestQFTCircuit(number, bits_per_number)

            start_time = time.time()
            task = device.run(test_circuit, shots=64)
            end_time = time.time()

            print(task.result().measurement_counts)

            print(f"Took {end_time - start_time} seconds")

    def testQFTInverseQFTIdempotent(device):
        for number in range(2 ** bits_per_number):
            test_circuit, number = buildTestQFTCircuit(number, bits_per_number)

            addToCircuit(test_circuit, invertInstructions(QFT(number).instructions))

            start_time = time.time()
            task = device.run(test_circuit, shots=64)
            end_time = time.time()

            print(task.result().measurement_counts)

            print(f"Took {end_time - start_time} seconds")

    from braket.devices import LocalSimulator

    global device
    device = LocalSimulator()

    testAllNumericQFT(device)
    testQFTInverseQFTIdempotent(device)
