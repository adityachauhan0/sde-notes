
#!/bin/bash
source venv/bin/activate
git add .
git commit -m "$1"
git push
mkdocs gh-deploy --clean
