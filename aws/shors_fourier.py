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
Method `FourierAdder` represents an adder sub-circuit in Fourier space.

Args:
    control_bits (List[int]): The first (pre-determined) value to be added.

    target (QubitSetInput): The second addend to be added to, in Fourier
                            space.  Stores the resulting addition of the
                            control_bits to the target, in Fourier space.
"""
def FourierAdder(
        control_bits: List[int],
        target: QubitSetInput
) -> Circuit:
    circuit = Circuit()

    two_pi_radians = 2 * math.pi
    num_qubits = len(target)
    for target_idx in range(num_qubits):
        for control_idx in range(0, target_idx + 1):
            if control_bits[control_idx] > 0:
                addToCircuit(circuit, Gate.PhaseShift.phaseshift(
                        target[target_idx],
                        two_pi_radians / (2 ** (target_idx - control_idx + 1))
                ))

    return circuit

"""
Method `CFourierAdder` represents an adder sub-circuit in Fourier space.

Args:
    control (QubitInput): The control qubit for enabling the circuit.

    control_bits (List[int]): The first (pre-determined) value to be added.

    target (QubitSetInput): The second addend to be added to, in Fourier
                            space.  Stores the resulting addition of the
                            control_bits to the target, in Fourier space.
"""
def CFourierAdder(
        control: QubitInput,
        control_bits: List[int],
        target: QubitSetInput
) -> Circuit:
    circuit = Circuit()

    two_pi_radians = 2 * math.pi
    for target_idx in range(len(target)):
        for control_idx in range(0, target_idx + 1):
            if control_bits[control_idx] > 0:
                addToCircuit(circuit, Gate.CPhaseShift.cphaseshift(
                        control,
                        target[target_idx],
                        two_pi_radians / (2 ** (target_idx - control_idx + 1))
                ))

    return circuit

"""
Method `CFourierModAdder` represents an adder sub-circuit in Fourier space,
performing the addition modulo a pre-determined number.

Args:
    control_qubit1 (QubitInput): The first control qubit for enabling the
                                 circuit.

    control_qubit2 (QubitInput): The second control qubit for enabling the
                                 circuit.

    control_qubit_internal (QubitInput): An intermediate qubit for combining
                                         control_qubit1 and control_qubit2
                                         into a single value for controlling
                                         CFourierAdder circuits.

    control_bits (List[int]): A list of int's representing the bit values of
                              the (pre-determined) addend.  These bits should
                              not be in Fourier space.

    target (QubitSetInput): The second addend to be added to, in Fourier
                            space.  Stores the resulting addition of the
                            control_bits to the target, in Fourier space.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the (pre-determined) modulus.  These bits should
                              not be in Fourier space.

    modulus_overflow_qubit (QubitInput): The value representing overflow of
                                         the sum over the modulus.  This qubit
                                         should be initialized to 0.  This
                                         method will return this qubit's value
                                         to 0 as well.
"""
def CFourierModAdder(
        control_qubit1: QubitInput,
        control_qubit2: QubitInput,
        control_qubit_internal: QubitInput,
        control_bits: List[int],
        target: QubitSetInput,
        modulus_bits: List[int],
        modulus_overflow_qubit: QubitInput
) -> Circuit:
    circuit = Circuit()

    addToCircuit(circuit, Gate.CCNot.ccnot(
            control_qubit1,
            control_qubit2,
            control_qubit_internal
    ))

    addToCircuit(circuit, CFourierAdder(
            control_qubit_internal,
            control_bits,
            target
    ))

    addToCircuit(circuit, invertInstructions(FourierAdder(
            modulus_bits,
            target
    ).instructions))

    addToCircuit(circuit, invertInstructions(QFT(target).instructions))

    target_first_qubit = target[-1]
    addToCircuit(circuit, Gate.CNot.cnot(
            target_first_qubit,
            modulus_overflow_qubit
    ))

    addToCircuit(circuit, QFT(target))

    addToCircuit(circuit, CFourierAdder(
            modulus_overflow_qubit,
            modulus_bits,
            target
    ))

    addToCircuit(circuit, invertInstructions(CFourierAdder(
            control_qubit_internal,
            control_bits,
            target
    ).instructions))

    addToCircuit(circuit, invertInstructions(QFT(target).instructions))

    addToCircuit(circuit, Gate.X.x(target_first_qubit))

    addToCircuit(circuit, Gate.CNot.cnot(
            target_first_qubit,
            modulus_overflow_qubit
    ))

    addToCircuit(circuit, Gate.X.x(target_first_qubit))

    addToCircuit(circuit, QFT(target))

    addToCircuit(circuit, CFourierAdder(
            control_qubit_internal,
            control_bits,
            target
    ))

    addToCircuit(circuit, Gate.CCNot.ccnot(
            control_qubit1,
            control_qubit2,
            control_qubit_internal
    ))

    return circuit

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

"""
Method `CFourierModMultiplier` represents a multiplier sub-circuit in Fourier
space, performing the addition modulo a pre-determined number.

Args:
    control_qubit (QubitInput): The control qubit for enabling the circuit.

    control_qubit_internal (QubitInput): An intermediate qubit required for
                                         the CFourierModAdder circuit.

    multiplier (QubitSetInput): The qubits representing the multiplier.

    multiplicand_int (int): The integer value of the multiplicand.

    target (QubitSetInput): The qubits to add the product to, if the
                            control_qubit is enabled.  If control_qubit is 1,
                            the value of these qubits after this circuit equals
                            (target + multiplier * control_int) % modulus_int,
                            where control_int and modulus_int are the integral
                            values of control_bits and modulus_bits.  If
                            control_qubit is 0, the value of these qubits
                            after this circuit remains unchanged.

    modulus_int (int): The integer value of the modulus.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the (pre-determined) modulus.  These bits should
                              not be in Fourier space.

    modulus_overflow_qubit (QubitInput): The value representing overflow of
                                         the multiplication over the modulus
                                         required for the CFourierModAdder
                                         circuit.  This qubit should be
                                         initialized to 0.  This method will
                                         return this qubit's value to 0 as
                                         well.
"""
def CFourierModMultiplier(
        control_qubit: QubitInput,
        control_qubit_internal: QubitInput,
        multiplier: QubitSetInput,
        multiplicand_int: int,
        product: QubitSetInput,
        modulus_int: int,
        modulus_bits: List[int],
        modulus_overflow_qubit: QubitInput
) -> Circuit:
    circuit = Circuit()

    addToCircuit(circuit, QFT(product))

    shift_factor = 1
    bits_per_number = len(multiplier)
    for control_idx in range(bits_per_number):
        multiplicand_mod_int = (shift_factor * multiplicand_int) % modulus_int
        multiplicand_mod_bits = getNumberAsBits(multiplicand_mod_int, bits_per_number)

        addToCircuit(circuit, CFourierModAdder(
                control_qubit,
                multiplier[control_idx],
                control_qubit_internal,
                multiplicand_mod_bits,
                product,
                modulus_bits,
                modulus_overflow_qubit
        ))

        shift_factor = 2 * shift_factor

    addToCircuit(circuit, invertInstructions(QFT(product).instructions))

    return circuit

"""
Method `CFourierModExponentiation` represents an exponentiation sub-circuit in
Fourier space, performing the exponentiation modulo a pre-determined number.

Args:
    exponent (QubitSetInput): The qubits representing the exponent.

    base_int (int): The integer value of the base to be exponentiated.

    control_qubit_internal (QubitInput): An intermediate qubit required for
                                         the CFourierModAdder circuit.

    multiplier (QubitSetInput): The qubits representing the multiplier
                                required for the CFourierModMultiplier
                                circuit.  These qubits should be set to 1
                                before this circuit.

    power (QubitSetInput): The qubits to add the exponentiation to.  These
                           qubits should be set to 0 before this circuit.
                           The value of these qubits after this circuit
                           equals (base_int ** exponent_int) % modulus_int.

    modulus_int (int): The integer value of the modulus.

    modulus_bits (List[int]): A list of int's representing the bit values of
                              the (pre-determined) modulus.  These bits should
                              not be in Fourier space.

    modulus_overflow_qubit (QubitInput): The value representing overflow of
                                         the exponentiation over the modulus
                                         required for the CFourierModAdder
                                         circuit.  This qubit should be
                                         initialized to 0.  This method will
                                         return this qubit's value to 0 as
                                         well.
"""
def CFourierModExponentiation(
        exponent: QubitSetInput,
        base_int: int,
        control_qubit_internal: QubitInput,
        multiplier: QubitSetInput,
        power: QubitSetInput,
        modulus_int: int,
        modulus_bits: List[int],
        modulus_overflow_qubit: QubitInput
) -> Circuit:
    circuit = Circuit()

    bits_per_number = len(exponent)
    for exponent_idx in range(bits_per_number):
        base_mod_int = (base_int ** (2 ** exponent_idx)) % modulus_int

        addToCircuit(circuit, CFourierModMultiplier(
                exponent[exponent_idx],
                control_qubit_internal,
                multiplier,
                base_mod_int,
                power,
                modulus_int,
                modulus_bits,
                modulus_overflow_qubit
        ))

        for multiplier_idx in range(len(multiplier)):
            addToCircuit(circuit, Gate.CSwap.cswap(
                    exponent[exponent_idx],
                    multiplier[multiplier_idx],
                    power[multiplier_idx]
            ))

        base_inv_mod_int = modInverse(base_mod_int, modulus_int)
        addToCircuit(circuit, invertInstructions(CFourierModMultiplier(
                exponent[exponent_idx],
                control_qubit_internal,
                multiplier,
                base_mod_int,
                power,
                modulus_int,
                modulus_bits,
                modulus_overflow_qubit
        ).instructions))

    return circuit
