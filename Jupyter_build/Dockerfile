ARG JUPBC=infotrend/define
FROM ${JUPBC}

RUN pip3 install ipywidgets \
    && jupyter labextension enable widgetsnbextension

EXPOSE 8888

# Using a default token 
SHELL ["/usr/bin/bash", "-c"]
CMD ["/usr/local/bin/python", "/usr/local/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token='iti'"]
