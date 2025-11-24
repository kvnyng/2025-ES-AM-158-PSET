# Workspace-specific .zshrc for harvard-es158-upkie conda environment

# Source user's home .zshrc if it exists (to preserve personal settings)
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

# Source conda initialization (if not already sourced)
if [ -f /Users/kvnyng/miniconda3/etc/profile.d/conda.sh ]; then
    source /Users/kvnyng/miniconda3/etc/profile.d/conda.sh
fi

# Activate the conda environment
conda activate harvard-es158-upkie 2>/dev/null || true

