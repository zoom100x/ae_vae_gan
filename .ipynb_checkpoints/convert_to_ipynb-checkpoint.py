import nbformat

# Path to the Python script with notebook content
script_path = "/home/ubuntu/Generative-Models-AE-VAE-GAN/Generative_Models_content.py"
# Path to save the output Jupyter notebook
notebook_path = "/home/ubuntu/Generative-Models-AE-VAE-GAN/Generative_Models.ipynb"

# Create a new notebook object
nb = nbformat.v4.new_notebook()

with open(script_path, "r") as f:
    lines = f.readlines()

current_cell_type = None # Can be 'code' or 'markdown'
current_cell_content = []

for line in lines:
    stripped_line = line.strip()

    if stripped_line == """""": # Triple quotes indicate a switch in cell type or end of a markdown block
        if current_cell_type == "markdown":
            # End of a markdown block
            if current_cell_content:
                nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(current_cell_content).strip()))
            current_cell_content = []
            current_cell_type = "code" # Assume next block is code unless another """ appears immediately
        else:
            # This is the start of a markdown block or an empty markdown block indicator
            # If there was preceding code, save it first
            if current_cell_content:
                nb.cells.append(nbformat.v4.new_code_cell("\n".join(current_cell_content).strip()))
            current_cell_content = []
            current_cell_type = "markdown"
    else:
        if current_cell_type == "markdown":
            current_cell_content.append(line.rstrip("\n")) # Keep original line breaks for markdown
        else: # Default to code cell if not in markdown or if current_cell_type is None (start of file)
            if current_cell_type is None: # Handle the very first lines as code by default
                current_cell_type = "code"
            current_cell_content.append(line.rstrip("\n"))

# Add any remaining content from the last cell
if current_cell_content:
    if current_cell_type == "markdown":
        nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(current_cell_content).strip()))
    elif current_cell_type == "code":
        # Ensure that initial empty lines or comments before the first """ are treated as code
        code_to_add = "\n".join(current_cell_content).strip()
        if code_to_add: # Only add if there's actual content
             nb.cells.append(nbformat.v4.new_code_cell(code_to_add))

# Save the notebook
with open(notebook_path, "w") as f:
    nbformat.write(nb, f)

print(f"Notebook saved to {notebook_path}")

