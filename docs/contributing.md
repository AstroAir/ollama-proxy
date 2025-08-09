# Contributing to Ollama Proxy

We welcome contributions from the community! Whether you're fixing a bug, adding a new feature, or improving the documentation, your help is greatly appreciated.

## How to Contribute

### Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

    ```bash
    git clone https://github.com/your-username/ollama-proxy.git
    cd ollama-proxy
    ```

3. **Set up a virtual environment** and install the dependencies, including the test dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[test]"
    ```

### Making Changes

1. **Create a new branch** for your feature or bug fix:

    ```bash
    git checkout -b my-new-feature
    ```

2. **Make your changes** to the code or documentation.
3. **Write tests** for any new features or bug fixes. We use `pytest` for testing.
4. **Run the tests** to ensure that everything is working correctly:

    ```bash
    pytest
    ```

5. **Format your code** using a code formatter if necessary. This project uses `black` and `isort`.

### Submitting Your Contribution

1. **Commit your changes** with a clear and descriptive commit message:

    ```bash
    git commit -m "feat: Add support for a new API endpoint"
    ```

2. **Push your branch** to your fork on GitHub:

    ```bash
    git push origin my-new-feature
    ```

3. **Open a pull request** from your branch to the `main` branch of the original repository.
4. **Provide a detailed description** of your changes in the pull request.

## Coding Guidelines

- **Follow the existing code style.**
- **Write clear and concise comments** where necessary.
- **Ensure your code is well-tested.**
- **Update the documentation** if you are adding or changing a feature.

## Reporting Bugs

If you find a bug, please open an issue on the GitHub repository. Include the following information in your report:

- A clear and descriptive title.
- A detailed description of the bug.
- Steps to reproduce the bug.
- Any relevant logs or error messages.

Thank you for contributing to the Ollama Proxy project!
