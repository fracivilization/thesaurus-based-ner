docker-compose up -d
docker-compose python run "poetry config virtualenvs.in-project true && docker-compose python poetry install --no-dev --no-interaction"