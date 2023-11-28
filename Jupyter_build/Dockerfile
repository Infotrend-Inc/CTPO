ARG JUPBC
FROM ${JUPBC}

EXPOSE 8888

# Adding a default token 
CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token='iti'
