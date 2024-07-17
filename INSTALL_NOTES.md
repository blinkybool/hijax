```bash
# Install python=3.12
pyenv install 3.12
pyenv local 3.12
# setup virtual env
python -m venv hijax.venv
source hijax.venv/bin/activate
# install jax/jaxlib/jax-metal with working versions
pip install jax==0.4.26 jaxlib==0.4.26
pip install jax-metal==0.1.0
# Test it works 
python -c 'import jax; print(jax.numpy.arange(10))' 
```