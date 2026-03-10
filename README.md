# lancell
Multimodal cell database in LanceDB

## Setup

Requires Python 3.13 and a Rust toolchain.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install Python deps
uv sync

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the Rust extension
maturin develop --release
```
