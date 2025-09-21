import subprocess

# Get the output of pip freeze
output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")

# Filter out lines with local paths
cleaned_requirements = []
for line in output.split('\n'):
    if ' @ ' not in line and line:  # Exclude lines with local paths and empty lines
        cleaned_requirements.append(line)

# Write the cleaned requirements to a file
with open("../requirements.txt", "w") as file:
    file.write('\n'.join(cleaned_requirements))
