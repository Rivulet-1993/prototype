FROM registry.sensetime.com/spring-test/algorithm/spring-dispatcher:latest as dispatcher
FROM registry.sensetime.com/spring-test/algorithm/spring-docker:latest as dockerdefault
FROM registry.sensetime.com/spring-test/spring-infra/encrypt-tool:latest as encryptor


FROM registry.sensetime.com/spring-test/spring-infra/algorithm:latest

RUN mkdir -p /spring/configs/default

COPY --from=dockerdefault /spring/* /spring/configs/default/
COPY --from=dispatcher /spring/SpringDispatcher /spring/SpringDispatcher
COPY --from=encryptor /enc_tools /enc_tools

RUN ln -sf /bin/bash /bin/sh

# nart 因为每个算法对nart的依赖不同，所以没放到基础镜像中
RUN mkdir -p /spring_install/nart && cd /spring_install/nart && \
    wget http://spring.sensetime.com/pypi/packages/nart-0.2.4-py3-none-any.whl#md5=36abf3e779767c11587b17364c43a626 -O nart-0.2.4-py3-none-any.whl && \
    pip3 install ./nart-0.2.4-py3-none-any.whl && \
    rm -rf /spring_install/nart

# prototype 使用torchvision 0.3.0会报错。降个版本解决
RUN pip3 install torchvision==0.2.2

# prototype
ADD ./ /spring/prototype 
RUN cd /spring/prototype && \
    python setup.py bdist_wheel && \
    pip install --user dist/*.whl

# 加密训练代码
RUN cd /spring && \
    mv prototype prototype_bak && \
    cd prototype_bak && \
    find . -name '*' -type f |grep -v '.git'| while read line ;\
        do \
            mkdir -p $(dirname ../prototype/$line) ;\
            cp -r $line ../prototype/$line ;\
            if [[ "${line##*.}" == "py" ]]; then \
                /enc_tools/model_encrypt_only -encrypt -enc_file /enc_tools/custom_enc_key.bin $line ../prototype/$line ;\
            fi  \
        done && \
    cd - && \
    rm -rf prototype_bak

# 加密dispatcher代码
RUN cd /spring && \
    mv SpringDispatcher SpringDispatcher_bak && \
    cd SpringDispatcher_bak && \
    find . -name '*' -type f |grep -v '.git'|while read line; do     mkdir -p $(dirname ../SpringDispatcher/$line);     cp -r $line ../SpringDispatcher/$line;     if [[ "${line##*.}" == "py" ]]; then /enc_tools/model_encrypt_only -encrypt -enc_file /enc_tools/custom_enc_key.bin $line ../SpringDispatcher/$line;     fi; done && \
     cd - && \
     rm -rf SpringDispatcher_bak

# 添加预训练模型
RUN mkdir -p /spring/pretrain && \
    cd /spring/pretrain && \
    wget "http://file.intra.sensetime.com/f/35d8d88afa/?raw=1" -O resnet101-cls.pth.tar



# 清理
RUN rm -rf /enc_tools
