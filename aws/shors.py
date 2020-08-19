# AWS imports: Import Braket SDK modules
import boto3
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

# aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
# s3_folder = (f"braket-output-{aws_account_id}", "RIGETTI")
# device = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-8")
device = LocalSimulator()

###############################################################################
#                                                                               
#                                Gates & Circuits                               
#                                                                               
###############################################################################
"""
Method `Carry` represents a carry sub-circuit for an adder.

Args:
    overflow_prev (QubitInput): The previous carry value.  This qubit is not
                                modified by this circuit.

    control1 (QubitInput): The first value to be added.  This qubit is not
                           modified by this circuit.

    control2 (QubitInput): The second value to be added.  Stores the resulting
                           sum, equal to overflow_prev XOR control1 XOR
                           control2.

    target (QubitInput): Stores the resulting overflow/carry value,
                         equal to (overflow_prev + control1 + control2) > 1
"""
def Carry(
        overflow_prev: QubitInput,
        control1: QubitInput,
        control2: QubitInput,
        target: QubitInput
) -> Circuit:
    circuit = Circuit()
    addToCircuit(circuit, Gate.CCNot.ccnot(control1, control2, target))
    addToCircuit(circuit, Gate.CNot.cnot(control1, control2))
    addToCircuit(circuit, Gate.CCNot.ccnot(overflow_prev, control2, target))
    return circuit

"""
Method `Sum` represents a sum sub-circuit for an adder.

Args:
    overflow_prev (QubitInput): The previous input value.  This qubit is not
                                modified by this circuit.

    control (QubitInput): The first value to be added.  This qubit is not
                          modified by this circuit.

    target (QubitInput): The second value to be added.  Stores the resulting
                         sum value, equal to overflow_prev XOR control XOR
                         target.
"""
def Sum(
        overflow_prev: QubitInput,
        control: QubitInput,
        target: QubitInput
) -> Circuit:
    circuit = Circuit()
    addToCircuit(circuit, Gate.CNot.cnot(control, target))
    addToCircuit(circuit, Gate.CNot.cnot(overflow_prev, target))
    return circuit

"""
Method `Adder` represents an adder with a deterministic input for an added.

Args:
    control (QubitSetInput): The first value representing an addend.  These
                             qubits are not modified by this circuit.

    target (QubitSetInput): The second value representing an addend.  The sum
                            of the control and target addends are stored in
                            these qubits.

    carry (QubitInput): Additional qubits to hold carry/overflow values. These
                        values should be initialized to 0.  This method will
                        return these values to 0 as well.  Should have one
                        one more qubit than `target` within its set.
"""
def Adder(
        control: QubitSetInput,
        target: QubitSetInput,
        carry: QubitSetInput
) -> Circuit:
    circuit = Circuit()
    for idx in range(len(target)):
        target_qubit = target[idx]
        control_qubit = control[idx]
        carry_qubit_prev = carry[idx]
        carry_qubit = carry[idx + 1]

        addToCircuit(circuit, Carry(
                carry_qubit_prev,
                control_qubit,
                target_qubit,
                carry_qubit
        ))

    target_qubit = target[-1]
    control_qubit = control[-1]
    carry_qubit_prev = carry[-2]

    addToCircuit(circuit, Gate.CNot.cnot(control_qubit, target_qubit))
    addToCircuit(circuit, Sum(
            carry_qubit_prev,
            control_qubit,
            target_qubit
    ))

    for idx in range(len(target) - 2, -1, -1):
        target_qubit = target[idx]
        control_qubit = control[idx]
        carry_qubit_prev = carry[idx]
        carry_qubit = carry[idx + 1]

        addToCircuit(circuit, invertInstructions(Carry(
                carry_qubit_prev,
                control_qubit,
                target_qubit,
                carry_qubit
        ).instructions))

        addToCircuit(circuit, Sum(
                carry_qubit_prev,
                control_qubit,
                target_qubit
        ))

    return circuit

"""
Method `ModAdder` represents a modular adder with a deterministic input for an
added, which returns the remainder of the sum modulo a deterministic modulus.

Args:
    control (QubitSetInput): The first value representing an addend.  These
                             qubits are not modified by this circuit.

    target (QubitSetInput): The second value representing an addend.  The sum
                            of the `control` and `target` addends are stored in
                            these qubits.

    carry (QubitInput): Additional qubits to hold carry/overflow values.
                        These values should be initialized to 0.  This method
                        will return these values to 0 as well.  Should have
                        one more qubit than `target` within its set.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the modulus.

    modulus (QubitSetInput): The value representing the modulus.  These qubits
                             should already be set to the bit value
                             representation of the modulus.  These qubits are
                             returned to their original state by this circuit.

    modulus_overflow_qubit (QubitInput): The value representing overflow of the
                                         sum over the modulus.  This qubit
                                         should be initialized to 0.  This
                                         method will return this qubit's value 
                                         to 0 as well.
"""
def ModAdder(
        control: QubitSetInput,
        target: QubitSetInput,
        carry: QubitSetInput,
        modulus_bits: List[int],
        modulus: QubitSetInput,
        modulus_overflow_qubit: QubitInput
) -> Circuit:
    circuit = Circuit()
    addToCircuit(circuit, Adder(control, target, carry))
    addToCircuit(circuit, invertInstructions(Adder(
            modulus,
            target,
            carry
    ).instructions))

    carry_qubit = carry[-1]
    addToCircuit(circuit, Gate.X.x(carry_qubit))
    addToCircuit(circuit, Gate.CNot.cnot(carry_qubit, modulus_overflow_qubit))
    addToCircuit(circuit, Gate.X.x(carry_qubit))

    for idx in range(len(modulus_bits)):
        modulus_bit = modulus_bits[idx]
        if modulus_bit > 0:
            addToCircuit(circuit, Gate.CNot.cnot(
                    modulus_overflow_qubit,
                    modulus[idx]
            ))

    addToCircuit(circuit, Adder(modulus, target, carry))

    for idx in range(len(modulus_bits)):
        modulus_bit = modulus_bits[idx]
        if modulus_bit > 0:
            addToCircuit(circuit, Gate.CNot.cnot(
                    modulus_overflow_qubit,
                    modulus[idx]
            ))

    addToCircuit(circuit, invertInstructions(Adder(
            control,
            target,
            carry
    ).instructions))

    addToCircuit(circuit, Gate.CNot.cnot(
            carry_qubit,
            modulus_overflow_qubit
    ))

    addToCircuit(circuit, Adder(control, target, carry))

    return circuit

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
                test_circuit = buildTestModAdderCircuit(addend2, addend1, modulus, bits_per_number)

                start_time = time.time()
                task = device.run(test_circuit, shots=1)
                end_time = time.time()

                for key in task.result().measurement_counts:
                    result = getNumberFromBitString(key, bits_per_number, bits_per_number)
                    expected = ((addend1 + addend2) % modulus)
                    print(f"Equation: ({addend1} + {addend2}) % {modulus}")
                    print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

                print(f"Took {end_time - start_time} seconds")

"""
Method `ModMultiplier` represents a modular multiplier with a pre-determined
input for an multiplicand, which returns the remainder of the multiplication
modulo a pre-determined modulus.

Args:
    control_qubit (QubitInput): The qubit representing whether the
                                multiplication should occur.  A value of 1 sets
                                the `product` gate to an effective value of
                                (multiplier * multiplicand_int) % modulus_int,
                                and a value of 0 sets the `product` gate to an
                                effective value of multiplier.

    multiplier (QubitSetInput): The first value representing the multiplier.
                                These qubits are potentially swapped with the
                                product qubits.

    product (QubitSetInput): The second value representing the multiplicand.
                             The multiplication of the `multiplier` and `product`
                             multiplicands are stored in these qubits.

    carry (QubitInput): Additional qubits to hold carry/overflow values.
                        These values should be initialized to 0.  This method
                        will return these values to 0 as well.  Should have
                        one more qubit than `product` within its set.

    modulus_int (int): The integer value of the modulus.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the modulus.

    modulus (QubitSetInput): The value representing the modulus.  These qubits
                             should already be set to the bit value
                             representation of the modulus.  These qubits are
                             returned to their original state by this circuit.

    modulus_overflow_qubit (QubitInput): The value representing overflow of the
                                         sum over the modulus.  This qubit
                                         should be initialized to 0.  This
                                         method will return this qubit's value 
                                         to 0 as well.

    multiplicand_int (int): An integer value of the non-variable multiplicand.

    multiplicand (QubitSetInput): The qubits representing the non-variable
                                  multiplicand.  These values should be
                                  initialized to 0.  This method will return
                                  these values to 0 as well.
"""
def ModMultiplier(
        control_qubit: QubitInput,
        multiplier: QubitSetInput,
        product: QubitSetInput,
        carry: QubitSetInput,
        modulus_int: int,
        modulus_bits: List[int],
        modulus: QubitSetInput,
        modulus_overflow_qubit: QubitInput,
        multiplicand_int: int,
        multiplicand: QubitSetInput
) -> Circuit:
    circuit = Circuit()

    shift_factor = 1
    bits_per_number = len(multiplier)
    for multiplier_idx in range(bits_per_number):
        multiplicand_mod_int = (multiplicand_int * shift_factor) % modulus_int
        multiplicand_mod_bits = getNumberAsBits(multiplicand_mod_int, bits_per_number)

        for multiplicand_mod_idx in range(len(multiplicand_mod_bits)):
            multiplicand_mod_bit = multiplicand_mod_bits[multiplicand_mod_idx]
            if multiplicand_mod_bit > 0:
                addToCircuit(circuit, Gate.CCNot.ccnot(
                        control_qubit,
                        multiplier[multiplier_idx],
                        multiplicand[multiplicand_mod_idx]
                ))

        addToCircuit(circuit, ModAdder(
                multiplicand,
                product,
                carry,
                modulus_bits,
                modulus,
                modulus_overflow_qubit
        ))

        for multiplicand_mod_idx in range(len(multiplicand_mod_bits)):
            multiplicand_mod_bit = multiplicand_mod_bits[multiplicand_mod_idx]
            if multiplicand_mod_bit > 0:
                addToCircuit(circuit, Gate.CCNot.ccnot(
                        control_qubit,
                        multiplier[multiplier_idx],
                        multiplicand[multiplicand_mod_idx]
                ))

        shift_factor = 2 * shift_factor

    addToCircuit(circuit, Gate.X.x(control_qubit))

    for multiplier_idx in range(len(multiplier)):
        addToCircuit(circuit, Gate.CCNot.ccnot(
                control_qubit,
                multiplier[multiplier_idx],
                product[multiplier_idx]
        ))

    addToCircuit(circuit, Gate.X.x(control_qubit))

    return circuit

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

# Testing ModExponentiation
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

"""
Method `ModExponentiation` represents a modular exponentiation with a
pre-determined input for a base, which returns the remainder of the
exponentiation modulo a pre-determined modulus.

Args:
    multiplier (QubitSetInput): The value representing the parital
                                exponentiation. These values should be
                                initialized to 0.  This method will return
                                these values to 0 as well.

    power (QubitSetInput): The value representing the result of the operation,
                           equivalent to (base ^ exponent) % modulus.  These
                           values should be initialized to 1.

    carry (QubitInput): Additional qubits to hold carry/overflow values.
                        These values should be initialized to 0.  This method
                        will return these values to 0 as well.  Should have
                        one more qubit than `power` within its set.

    modulus_int (int): The integer value of the modulus.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the modulus.

    modulus (QubitSetInput): The value representing the modulus.  These qubits
                             should already be set to the bit value
                             representation of the modulus.  These qubits are
                             returned to their original state by this circuit.

    modulus_overflow_qubit (QubitInput): The value representing overflow of the
                                         sum over the modulus.  This qubit
                                         should be initialized to 0.  This
                                         method will return this qubit's value 
                                         to 0 as well.

    base_int (int): An integer value of the non-variable base.

    base (QubitSetInput): The qubits representing the non-variable
                          base.  These values should be
                          initialized to 0.  This method will return
                          these values to 0 as well.

    exponent (QubitSetInput): The value representing the exponent. These
                              qubits are not modified by this circuit.
"""
def ModExponentiation(
        multiplier: QubitSetInput,
        product: QubitSetInput,
        carry: QubitSetInput,
        modulus_int: int,
        modulus_bits: List[int],
        modulus: QubitSetInput,
        modulus_overflow_qubit: QubitInput,
        base_int: int,
        base: QubitSetInput,
        exponent: QubitSetInput
) -> Circuit:
    circuit = Circuit()

    current_multiplier = multiplier
    current_product = product

    for exponent_idx in range(len(exponent)):
        base_mod_int = (base_int ** (2 ** exponent_idx)) % modulus_int
        addToCircuit(circuit, ModMultiplier(
                exponent[exponent_idx],
                current_multiplier,
                current_product,
                carry,
                modulus_int,
                modulus_bits,
                modulus,
                modulus_overflow_qubit,
                base_mod_int,
                base
        ))

        next_product = current_multiplier
        current_multiplier = current_product
        current_product = next_product

        base_inv_mod_int = modInverse(base_mod_int, modulus_int)
        addToCircuit(circuit, invertInstructions(ModMultiplier(
                exponent[exponent_idx],
                current_multiplier,
                current_product,
                carry,
                modulus_int,
                modulus_bits,
                modulus,
                modulus_overflow_qubit,
                base_inv_mod_int,
                base
        ).instructions))

    return circuit

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
        task = device.run(test_circuit, shots=1)
        end_time = time.time()

        print(task.result().measurement_counts)

        for key in task.result().measurement_counts:
            result = getNumberFromBitString(key, bits_per_number, bits_per_number)
            expected = ((base ** exponent) % modulus)
            print(f"Equation: ({base} ** {exponent}) % {modulus}")
            print(f"Result: {result}, Expected: {expected}, Accurate: {result == expected}")

        print(f"Took {end_time - start_time} seconds")

bits_per_number = 3

# Testing ModAdder
# test_circuit = buildTestModAdderCircuit(2, 3, 7, bits_per_number)
# testAllInputsModAdder(bits_per_number)

# Testing ModMultiplier
# test_circuit = buildTestModMultiplierCircuit(2, 3, 7, bits_per_number)
testAllInputsModMultiplier(bits_per_number)

# Testing ModExponentiation
# testAllExponentsModExponentiation(2, 7, bits_per_number)
