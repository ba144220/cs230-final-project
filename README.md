# cs230-final-project

For the following instructions, `mamba` and `conda` are interchangeable. `mamba` is recommended.

# Environment
## First Time Setup
1. Create the environment:
```bash
mamba env create -f environment.yaml
```
2. Activate the environment:
```bash
mamba activate cs230
```
If you want to deactivate the environment, run `mamba deactivate`.

3. Create a `.env` file based on `.env.example` and put your environment variables in it.

## Development
Before developing, you should activate the environment:
```bash
mamba activate cs230
```
You can deactivate the environment by running `mamba deactivate`.

## Installing New Packages

1. Add the package to `environment.yaml` and run
```bash
mamba env update -f environment.yaml
```
If you use `pip` to install packages, add them to `environment.yaml` under `pip` section.
**YOU SHOULD NOT INSTALL NEW PACKAGES DIRECTLY THROUGH `conda`, `mamba`, or `pip`.**


# Conventions

## Commit Messages

Use the following format for commit messages:
```
<type>: <description>
```
The type could be one of the following:
- `fix`: A bug fix
- `docs`: Changes to the documentation
- `style`: Formatting, missing semi colons, etc; no code changes
- `chore`: Updating build configuration, development scripts, etc.
- `experiment`: Experimenting with the codebase
- `modeling`: Changes to the modeling codebase
- `dataset`: Changes to the dataset codebase

## Branching

Branch name should be in the following format: `<type>/description-of-the-branch`

For the type, we use the following naming convention:
- `main`: The main branch is the production branch.
- `dev`: The development branch is the branch where the development happens.
- `experiment/`: The experiment branch is the branch where the new experiment development happens.
- `modeling/`: The modeling branch is the branch where the new modeling code development happens.

## File/Folder Naming Conventions
* `all_lowercase_with_underline`
* Put production code in `src/`
* Do experiments in `notebooks/`

## Variable Naming Conventions
* Functions
    * `all_lowercase_with_underline`
* Classes
    * `CamelCase`





# Dataset
## Dataset File Structure
* Store datasets at `./datasets`
* For splits, add `_train` or `_test` at the end of the folder name.
* File structure
    * A `qa.csv` file contain all the questions and their corresponding context and answer
        * question: str
        * answer: str
        * context: str (only the name of the table)
        * id: unique str
        * task: optional enum (“arithmetic” and “list-item”)
        * direction: optional enum (“row” or “col”)
        * size: optional tuple[int] 
    * A folder name `tables` that contains .csv, .html, and .table file of the table.
## Data Class
* Data class: `TableDataset`
    * Simple 
        * Output format can be on of .csv, .html, and .table
        * Output: `List[string]`
    * Hard
        * Output format can be on of .csv, .html, .table, 2D
        * Output: `Tensor` (num_of_samples, longest length)

## Self-generated Dataset Specifications
### General
#### Dataset Size
* Test set: TBD
* Train set: TBD
#### Table Sizes
All tables are N*N, N=4,6,8,10,12

#### Number of Questions
For each table, generate 4 questions for rows and columns. (Total 8 questions per table)

### Arithmetic Task
* Range of numbers: 1-10
* For arithmetic, just do min and max.

### Item-Listing Task
* Range of letters: A-Z
* Answers are separated by commas.
    * e.g. A, B, C, D


# Modeling

Here is the modules hierarchy of the Llama 3.2 model:
```Mermaid
flowchart TD
    LlamaForCausalLM --> LlamaModel
    LlamaModel --> LlamaDecoderLayer
    LlamaDecoderLayer --> LlamaAttention
```