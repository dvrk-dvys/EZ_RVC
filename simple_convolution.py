import random


def simple_convolution(input_data, kernel):
    kernel_size = len(kernel)
    output_size = len(input_data) - kernel_size + 1
    output = [0] * output_size

    for i in range(output_size):
        for j in range(kernel_size):
            output[i] += input_data[i + j] * kernel[j]

    return output


# Generate a random input sequence of length 10 with values between 0 and 9
input_sequence = [random.randint(0, 9) for _ in range(10)]
print(f"Input Sequence: {input_sequence}")

# Define a simple kernel of size 3
kernel = [0.5, 1, 0.5]
print(f"Kernel: {kernel}")

# Perform convolution
output_sequence = simple_convolution(input_sequence, kernel)
print(f"Output Sequence after Convolution: {output_sequence}")
