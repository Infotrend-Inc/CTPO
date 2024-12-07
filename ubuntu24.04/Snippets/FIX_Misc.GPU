# - Avoid "RPC failed; curl 56 GnuTLS" issue on some pulls, while keeping the system installed git
# - Some tools expect a "python" binary
# - Prepare ldconfig
RUN git config --global http.postBuffer 1048576000 \
  && mkdir -p /usr/local/bin && ln -s $(which python3) /usr/local/bin/python \
  && mkdir -p /usr/local/lib && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/usrlocallib.conf' && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Misc GPU fixes
RUN cd /usr/local \
  && if [ -e cuda ]; then if [ ! -e nvidia ]; then ln -s cuda nvidia; fi; fi \
  && tmp="/usr/local/cuda/extras/CUPTI/lib64" \
  && if [ -d $tmp ]; then \ 
        echo $tmp >> /etc/ld.so.conf.d/nvidia-cupti.conf; \
        ldconfig; \
        echo "***** CUPTI added to LD path"; \
     fi
