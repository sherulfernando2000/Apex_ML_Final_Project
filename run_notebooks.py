"""Execute all notebooks in sequence using nbconvert Python API."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

NOTEBOOKS = [
    'notebooks/1_eda_and_preprocessing.ipynb',
    'notebooks/2_supervised_modeling.ipynb',
    'notebooks/3_unsupervised_clustering.ipynb',
    'notebooks/4_mlops_deployment.ipynb',
]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_path in NOTEBOOKS:
    print(f'\n{"="*60}')
    print(f'Executing: {nb_path}')
    print(f'{"="*60}')
    
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute from the notebook's directory so relative paths work
        nb_dir = os.path.dirname(os.path.abspath(nb_path))
        ep.preprocess(nb, {'metadata': {'path': nb_dir}})
        
        # Save the executed notebook (overwrite original)
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f'SUCCESS: {nb_path}')
    except Exception as e:
        print(f'ERROR in {nb_path}: {e}')
        # Still save what we have
        try:
            with open(nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        except:
            pass
        # Don't stop - continue to next notebook if dependencies allow
        if nb_path == NOTEBOOKS[0]:
            print('CRITICAL: Notebook 1 failed. Stopping - other notebooks depend on it.')
            sys.exit(1)

print('\n' + '='*60)
print('All notebooks executed!')
print('='*60)
