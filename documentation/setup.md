# Set up

### Get repository

### Set up virtual environment

* Install [conda](https://docs.conda.io/en/latest/miniconda.html#macosx-installers) (take the `.sh` file)
    * **N.B.** If your terminal runs under zsh, you should [configure your zsh](https://www.freecodecamp.org/news/how-to-configure-your-macos-terminal-with-zsh-like-a-pro-c0ab3f3c1156/) and add `source .bash_profile` in your `.zshrc` file.

* Create environment : `conda env create -f conda.yaml`
* Activate environment : `conda activate foodcast`
* Add the environement to jupyter : `python -m ipykernel install --user --name=foodcast`

In case you want to remove the environment from jupyter : `jupyter kernelspec uninstall foodcast`

<[Précédent](../README.md) | [Suivant](data.md)>
