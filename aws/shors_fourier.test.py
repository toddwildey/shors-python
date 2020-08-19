# AWS imports: Import Braket SDK modules
import boto3
import math
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
    invertInstructions, \
    setCircuitBits, \
    getNumberAsBits, \
    getNumberFromBitString, \
    QFT

from shors_fourier import \
    FourierAdder, \
    CFourierAdder, \
    CFourierModAdder, \
    CFourierModMultiplier, \
    CFourierModExponentiation

device = LocalSimulator()

# Testing FourierAdder
def buildTestFourierAdderCircuit(control_int, number_int, bits_per_number):
    control_bits = getNumberAsBits(control_int, bits_per_number)
    number_bits = getNumberAsBits(number_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    number = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, number_bits, bit_index)
    bit_index += bits_per_number

    addToCircuit(circuit, QFT(number))
    addToCircuit(circuit, FourierAdder(control_bits, number))
    addToCircuit(circuit, invertInstructions(QFT(number).instructions))

    return circuit, number

def testAllInputsFourierAdder(bits_per_number):
    for control_int in range(2 ** bits_per_number):
        for addend_int in range(2 ** bits_per_number):
            test_circuit, number = buildTestFourierAdderCircuit(control_int, addend_int, bits_per_number)

            start_time = time.time()
            task = device.run(test_circuit, shots=1)
            end_time = time.time()

            print(task.result().measurement_counts)

            for key in task.result().measurement_counts:
                result = getNumberFromBitString(key, bits_per_number, 0)
                expected = (control_int + addend_int) % (2 ** bits_per_number)
                print(f"Equation: ({control_int} + {addend_int})")
                print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                assert(result == expected)

            print(f"Took {end_time - start_time} seconds")

# Testing CFourierAdder
def buildTestCFourierAdderCircuit(control_int, number_int, control_bit, bits_per_number):
    control_bits = getNumberAsBits(control_int, bits_per_number)
    number_bits = getNumberAsBits(number_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    number = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, number_bits, bit_index)
    bit_index += bits_per_number

    control_qubit = Qubit(bit_index)
    setCircuitBits(circuit, [ control_bit ], bit_index)
    bit_index += 1

    addToCircuit(circuit, QFT(number))
    addToCircuit(circuit, CFourierAdder(control_qubit, control_bits, number))
    addToCircuit(circuit, invertInstructions(QFT(number).instructions))

    return circuit, number

def testAllInputsCFourierAdder(bits_per_number):
    # Testing control bit enabled
    for control_int in range(2 ** bits_per_number):
        for addend_int in range(2 ** bits_per_number):
            test_circuit, number = buildTestCFourierAdderCircuit(control_int, addend_int, 1, bits_per_number)

            start_time = time.time()
            task = device.run(test_circuit, shots=1)
            end_time = time.time()

            print(task.result().measurement_counts)

            for key in task.result().measurement_counts:
                result = getNumberFromBitString(key, bits_per_number, 0)
                expected = (control_int + addend_int) % (2 ** bits_per_number)
                print(f"Equation: ({control_int} + {addend_int}) % ({2 ** bits_per_number})")
                print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                assert(result == expected)

            print(f"Took {end_time - start_time} seconds")

    # Testing control bit disabled
    for control_int in range(2 ** bits_per_number):
        for addend_int in range(2 ** bits_per_number):
            test_circuit, number = buildTestCFourierAdderCircuit(control_int, addend_int, 0, bits_per_number)

            start_time = time.time()
            task = device.run(test_circuit, shots=1)
            end_time = time.time()

            print(task.result().measurement_counts)

            for key in task.result().measurement_counts:
                result = getNumberFromBitString(key, bits_per_number, 0)
                expected = addend_int
                print(f"Equation: {addend_int}")
                print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                assert(result == expected)

            print(f"Took {end_time - start_time} seconds")

# Testing CFourierModAdder
def buildTestCFourierModAdderCircuit(
        control_int,
        number_int,
        modulus_int,
        control_bit_1,
        control_bit_2,
        bits_per_number
):
    bits_per_number = bits_per_number + 1
    control_bits = getNumberAsBits(control_int, bits_per_number)
    number_bits = getNumberAsBits(number_int, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    number = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, number_bits, bit_index)
    bit_index += bits_per_number

    control_qubit_1 = Qubit(bit_index)
    setCircuitBits(circuit, [ control_bit_1 ], bit_index)
    bit_index += 1

    control_qubit_2 = Qubit(bit_index)
    setCircuitBits(circuit, [ control_bit_2 ], bit_index)
    bit_index += 1

    control_qubit_internal = Qubit(bit_index)
    bit_index += 1

    modulus_overflow_qubit = Qubit(bit_index)
    bit_index += 1

    addToCircuit(circuit, QFT(number))
    addToCircuit(circuit, CFourierModAdder(
            control_qubit_1,
            control_qubit_2,
            control_qubit_internal,
            control_bits,
            number,
            modulus_bits,
            modulus_overflow_qubit
    ))
    addToCircuit(circuit, invertInstructions(QFT(number).instructions))

    return circuit, number

def testAllInputsCFourierModAdder(bits_per_number):
    # Testing control bit disabled
    for modulus_int in range(2, 2 ** bits_per_number):
        for control_int in range(modulus_int):
            for addend_int in range(modulus_int):
                test_circuit, number = buildTestCFourierModAdderCircuit(
                        control_int,
                        addend_int,
                        modulus_int,
                        0,
                        0,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = addend_int
                    print(f"Equation: {addend_int} % {modulus_int}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

                test_circuit, number = buildTestCFourierModAdderCircuit(
                        control_int,
                        addend_int,
                        modulus_int,
                        1,
                        0,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = addend_int
                    print(f"Equation: {addend_int} % {modulus_int}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

                test_circuit, number = buildTestCFourierModAdderCircuit(
                        control_int,
                        addend_int,
                        modulus_int,
                        0,
                        1,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = addend_int
                    print(f"Equation: {addend_int} % {modulus_int}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

    # Testing control bit enabled
    for modulus_int in range(2, 2 ** bits_per_number):
        for control_int in range(modulus_int):
            for addend_int in range(modulus_int):
                test_circuit, number = buildTestCFourierModAdderCircuit(
                        control_int,
                        addend_int,
                        modulus_int,
                        1,
                        1,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = (control_int + addend_int) % (modulus_int)
                    print(f"Equation: ({control_int} + {addend_int}) % ({modulus_int})")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

# Testing CFourierModMultiplier
def buildTestCFourierModMultiplierCircuit(
        multiplier_int,
        multiplicand_int,
        modulus_int,
        control_bit,
        bits_per_number
):
    bits_per_number = bits_per_number + 1
    multiplier_bits = getNumberAsBits(multiplier_int, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    product_int = 0
    product_bits = getNumberAsBits(product_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    product = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, product_bits, bit_index)
    bit_index += bits_per_number

    multiplier = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, multiplier_bits, bit_index)
    bit_index += bits_per_number

    control_qubit = Qubit(bit_index)
    setCircuitBits(circuit, [ control_bit ], bit_index)
    bit_index += 1

    control_qubit_internal = Qubit(bit_index)
    bit_index += 1

    modulus_overflow_qubit = Qubit(bit_index)
    bit_index += 1

    addToCircuit(circuit, CFourierModMultiplier(
            control_qubit,
            control_qubit_internal,
            multiplier,
            multiplicand_int,
            product,
            modulus_int,
            modulus_bits,
            modulus_overflow_qubit
    ))

    return circuit, product

def testAllInputsCFourierModMultiplier(bits_per_number):
    # Testing control bit disabled
    for modulus_int in range(2, 2 ** bits_per_number):
        for multiplier_int in range(modulus_int):
            for multiplicand_int in range(modulus_int):
                test_circuit, number = buildTestCFourierModMultiplierCircuit(
                        multiplier_int,
                        multiplicand_int,
                        modulus_int,
                        0,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = 0
                    print(f"Equation: 0 % {modulus_int}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

    # Testing control bit enabled
    for modulus_int in range(2, 2 ** bits_per_number):
        for multiplier_int in range(modulus_int):
            for multiplicand_int in range(modulus_int):
                test_circuit, number = buildTestCFourierModMultiplierCircuit(
                        multiplier_int,
                        multiplicand_int,
                        modulus_int,
                        1,
                        bits_per_number
                )

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, 0)
                    expected = (multiplier_int * multiplicand_int) % (modulus_int)
                    print(f"Equation: ({multiplier_int} * {multiplicand_int}) % ({modulus_int})")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")


# Testing CFourierModExponentiation
def buildTestCFourierModExponentiationCircuit(
        base_int,
        exponent_int,
        modulus_int,
        bits_per_number
):
    bits_per_number = bits_per_number + 1
    exponent_bits = getNumberAsBits(exponent_int, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    multiplier_int = 1
    multiplier_bits = getNumberAsBits(multiplier_int, bits_per_number)

    power_int = 0
    power_bits = getNumberAsBits(power_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    power = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, power_bits, bit_index)
    bit_index += bits_per_number

    multiplier = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, multiplier_bits, bit_index)
    bit_index += bits_per_number

    exponent = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, exponent_bits, bit_index)
    bit_index += bits_per_number

    control_qubit_internal = Qubit(bit_index)
    bit_index += 1

    modulus_overflow_qubit = Qubit(bit_index)
    bit_index += 1

    addToCircuit(circuit, CFourierModExponentiation(
            exponent,
            base_int,
            control_qubit_internal,
            multiplier,
            power,
            modulus_int,
            modulus_bits,
            modulus_overflow_qubit
    ))

    return circuit, exponent, multiplier

def testAllInputsCFourierModExponentiation(bits_per_number):
    for modulus_int in range(3, 2 ** bits_per_number, 2):
        for base_int in range(2, modulus_int):
            for exponent_int in range(modulus_int):
                expected = (base_int ** exponent_int) % (modulus_int)

                test_circuit, exponent, multiplier = buildTestCFourierModExponentiationCircuit(
                        base_int,
                        exponent_int,
                        modulus_int,
                        bits_per_number
                )

                bits_per_number = bits_per_number + 1

                device = LocalSimulator()

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                print(task.result().measurement_counts)

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, bits_per_number)
                    expected = (base_int ** exponent_int) % (modulus_int)
                    print(f"Equation: ({base_int} ** {exponent_int}) % ({modulus_int})")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")
                    assert(result == expected)

                print(f"Took {end_time - start_time} seconds")

# Testing FourierShors
def buildTestFourierShorsCircuit(
        base_int,
        modulus_int,
        bits_per_number
):
    bits_per_number = bits_per_number + 1
    exponent_bits = getNumberAsBits(0, bits_per_number)
    modulus_bits = getNumberAsBits(modulus_int, bits_per_number)

    multiplier_int = 7
    multiplier_bits = getNumberAsBits(multiplier_int, bits_per_number)

    power_int = 0
    power_bits = getNumberAsBits(power_int, bits_per_number)

    circuit = Circuit()

    bit_index = 0
    power = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, power_bits, bit_index)
    bit_index += bits_per_number

    multiplier = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    setCircuitBits(circuit, multiplier_bits, bit_index)
    bit_index += bits_per_number

    exponent = QubitSet([idx for idx in range(bit_index, bit_index + bits_per_number)])
    bit_index += bits_per_number

    control_qubit_internal = Qubit(bit_index)
    bit_index += 1

    modulus_overflow_qubit = Qubit(bit_index)
    bit_index += 1

    addToCircuit(circuit, invertInstructions(CFourierModExponentiation(
            exponent,
            base_int,
            control_qubit_internal,
            multiplier,
            power,
            modulus_int,
            modulus_bits,
            modulus_overflow_qubit
    ).instructions))

    addToCircuit(circuit, invertInstructions(QFT(exponent).instructions))

    return circuit, exponent, multiplier

def testFourierShors(bits_per_number):
    modulus_int = 15
    base_int = 2

    test_circuit, exponent, multiplier = buildTestFourierShorsCircuit(
            base_int,
            modulus_int,
            bits_per_number
    )

    bits_per_number = bits_per_number + 1

    start_time = time.time()
    task = device.run(test_circuit, shots=128)
    end_time = time.time()

    print(task.result().measurement_counts)

    for key in task.result().measurement_counts:
        result = getNumberFromBitString(key, bits_per_number - 1, 2 * bits_per_number)
        print(f"Result: {result}")

    print(f"Took {end_time - start_time} seconds")

bits_per_number = 4

# Testing FourierAdder
# testAllInputsFourierAdder(bits_per_number)

# Testing CFourierAdder
# testAllInputsCFourierAdder(bits_per_number)

# Testing CFourierModAdder
# testAllInputsCFourierModAdder(bits_per_number)

# Testing CFourierModMultiplier
# testAllInputsCFourierModMultiplier(bits_per_number)

# Testing CFourierModExponentiation
# testAllInputsCFourierModExponentiation(bits_per_number)

# Testing FourierShors
testFourierShors(bits_per_number)
