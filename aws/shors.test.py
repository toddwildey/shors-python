# AWS imports: Import Braket SDK modules
import time

from typing import Iterable, List

from braket.circuits import Circuit
from braket.circuits.angled_gate import AngledGate
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction
from braket.circuits.qubit import QubitInput, Qubit
from braket.circuits.qubit_set import QubitSet, QubitSetInput
from braket.devices import LocalSimulator
from braket.aws import AwsSession, AwsDevice

from utils import \
    modInverse, \
    addToCircuit, \
    invertInstruction, \
    invertInstructions, \
    setCircuitBits, \
    getNumberAsBits, \
    getNumberFromBitString, \
    QFT

from shors import \
    Carry, \
    Sum, \
    Adder, \
    ModAdder, \
    ModMultiplier, \
    ModExponentiation

device = LocalSimulator()

###############################################################################
#
#                                Tests
#
###############################################################################

def buildTestCarryCircuit(overflow_prev_int, control1_int, control2_int, bits_per_number):
    overflow_prev_bits = getNumberAsBits(overflow_prev_int, 1)
    control1_bits = getNumberAsBits(control1_int, 1)
    control2_bits = getNumberAsBits(control2_int, 1)

    circuit = Circuit()

    bit_index = 0
    overflow_prev = Qubit(bit_index)
    setCircuitBits(circuit, overflow_prev_bits, bit_index)
    bit_index += 1

    control1 = Qubit(bit_index)
    setCircuitBits(circuit, control1_bits, bit_index)
    bit_index += 1

    control2 = Qubit(bit_index)
    setCircuitBits(circuit, control2_bits, bit_index)
    bit_index += 1

    target = Qubit(bit_index)
    bit_index += 1

    circuit.add(Carry(overflow_prev, control1, control2, target))

    return circuit

def testAllInputsCarry(bits_per_number = 1):
    for overflow_prev in range(2):
        for control1 in range(2):
            for control2 in range(2):
                test_circuit = buildTestCarryCircuit(overflow_prev, control1, control2, bits_per_number)

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                for key in task.result().measurement_counts:
                    # Target is the 4th qubit (index 3)
                    carry_result = getNumberFromBitString(key, 1, 3)
                    # control2 is modified to store the sum
                    sum_result = getNumberFromBitString(key, 1, 2)

                    expected_carry = 1 if (overflow_prev + control1 + control2) > 1 else 0
                    expected_sum = (overflow_prev ^ control1 ^ control2)

                    print(f"Carry({overflow_prev}, {control1}, {control2})")
                    print(f"Carry Result: {carry_result}, Expected: {expected_carry}, Accurate: {carry_result == expected_carry}")
                    print(f"Sum Result: {sum_result}, Expected: {expected_sum}, Accurate: {sum_result == expected_sum}")

                print(f"Took {end_time - start_time} seconds")

def buildTestAdderCircuit(control_int, target_int, bits_per_number):
    control_bits = getNumberAsBits(control_int, bits_per_number)
    target_bits = getNumberAsBits(target_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    control = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, control_bits, bit_index)
    bit_index += bits_per_number

    target = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, target_bits, bit_index)
    bit_index += bits_per_number

    carry = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number + 1)])
    bit_index += bits_per_number + 1

    circuit.add(Adder(control, target, carry))

    return circuit

def testAllInputsAdder(bits_per_number = 3):
    for control in range(2 ** bits_per_number):
        for target in range(2 ** bits_per_number):
            test_circuit = buildTestAdderCircuit(control, target, bits_per_number)

            start_time = time.time()
            task = device.run(test_circuit, shots=1)
            end_time = time.time()

            for key in task.result().measurement_counts:
                result = getNumberFromBitString(key, bits_per_number, bits_per_number)
                expected = (control + target) % (2 ** bits_per_number)
                print(f"Equation: ({control} + {target})")
                print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

            print(f"Took {end_time - start_time} seconds")

def buildTestModAdderCircuit(addend1_int, addend2_int, modulus_int, bits_per_number):
    addend1_bits = getNumberAsBits(addend1_int, bits_per_number)
    addend2_bits = getNumberAsBits(addend2_int, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    addend1 = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, addend1_bits, bit_index)
    bit_index += bits_per_number

    addend2 = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, addend2_bits, bit_index)
    bit_index += bits_per_number

    carry = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number + 1)])
    bit_index += bits_per_number + 1

    modulus = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, modulus_bits, bit_index)
    bit_index += bits_per_number

    modulus_overflow = Qubit(bit_index)
    bit_index += 1

    circuit.add(ModAdder(addend1, addend2, carry, modulus_bits, modulus, modulus_overflow))

    return circuit

def testAllInputsModAdder(bits_per_number = 3):
    for modulus in range(3, (2 ** bits_per_number) - 1):
        for addend1 in range(modulus):
            for addend2 in range(modulus):
                test_circuit = buildTestModAdderCircuit(addend1, addend2, modulus, bits_per_number)

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, bits_per_number)
                    expected = ((addend1 + addend2) % modulus)
                    print(f"Equation: ({addend1} + {addend2}) % {modulus}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

                print(f"Took {end_time - start_time} seconds")

def buildTestModMultiplierCircuit(multiplicand_int, multiplier_int, modulus_int, bits_per_number):
    multiplier_bits = getNumberAsBits(multiplier_int, bits_per_number)

    product_int = 0
    product_bits = getNumberAsBits(product_int, bits_per_number)

    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    multiplier = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, multiplier_bits, bit_index)
    bit_index += bits_per_number

    product = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, product_bits, bit_index)
    bit_index += bits_per_number

    carry = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number + 1)])
    bit_index += bits_per_number + 1

    modulus = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, modulus_bits, bit_index)
    bit_index += bits_per_number

    modulus_overflow = Qubit(bit_index)
    bit_index += 1

    multiplicand = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    bit_index += bits_per_number

    multiplier_enabled = Qubit(bit_index)
    circuit.x(multiplier_enabled)
    bit_index += 1

    circuit.add(ModMultiplier(
            multiplier_enabled,
            multiplier,
            product,
            carry,
            modulus_int,
            modulus_bits,
            modulus,
            modulus_overflow,
            multiplicand_int,
            multiplicand
    ))

    return circuit

def testAllInputsModMultiplier(bits_per_number = 3):
    for modulus in range(3, (2 ** bits_per_number) - 1):
        for multiplier in range((2 ** bits_per_number) - 1):
            for multiplicand in range((2 ** bits_per_number) - 1):
                test_circuit = buildTestModMultiplierCircuit(multiplicand, multiplier, modulus, bits_per_number)

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, bits_per_number)
                    expected = ((multiplier * multiplicand) % modulus)
                    print(f"Equation: ({multiplier} * {multiplicand}) % {modulus}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

                print(f"Took {end_time - start_time} seconds")

def buildTestModExponentiationCircuit(base_int, exponent_int, modulus_int, bits_per_number):
    base_bits = getNumberAsBits(base_int, bits_per_number)
    exponent_bits = getNumberAsBits(exponent_int, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    multiplier_int = 1
    multiplier_bits = getNumberAsBits(multiplier_int, bits_per_number)

    product_int = 0
    product_bits = getNumberAsBits(product_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    multiplier = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, multiplier_bits, bit_index)
    bit_index += bits_per_number

    product = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, product_bits, bit_index)
    bit_index += bits_per_number

    carry = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number + 1)])
    bit_index += bits_per_number + 1

    modulus = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, modulus_bits, bit_index)
    bit_index += bits_per_number

    modulus_overflow = Qubit(bit_index)
    bit_index += 1

    base = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    bit_index += bits_per_number

    exponent = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, exponent_bits, bit_index)
    bit_index += bits_per_number

    circuit.add(ModExponentiation(
            multiplier,
            product,
            carry,
            modulus_int,
            modulus_bits,
            modulus,
            modulus_overflow,
            base_int,
            base,
            exponent
    ))

    return circuit

def testAllExponentsModExponentiation(base, modulus, bits_per_number):
    for exponent in range(2 ** bits_per_number):
        test_circuit = buildTestModExponentiationCircuit(base, exponent, modulus, bits_per_number)

        start_time = time.time()
        task = device.run(test_circuit, shots=1000)
        end_time = time.time()

        print(task.result().measurement_counts)

        for key in task.result().measurement_counts:
            result = getNumberFromBitString(key, bits_per_number, bits_per_number)
            expected = ((base ** exponent) % modulus)
            print(f"Equation: ({base} ** {exponent}) % {modulus}")
            print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

        print(f"Took {end_time - start_time} seconds")

if __name__ == '__main__':
    bits_per_number = 3

    # Testing `Carry`
    # Doesn't pass all tests, because it must be used in conjunction with a `Sum` circuit in an overall `Adder` circuit
    # testAllInputsCarry(1)

    # Testing `Adder`
    testAllInputsAdder(bits_per_number)

    # Testing ModAdder
    # testAllInputsModAdder(bits_per_number)

    # Testing ModMultiplier
    # testAllInputsModMultiplier(bits_per_number)

    # Testing ModExponentiation
    # testAllExponentsModExponentiation(2, 7, bits_per_number)
