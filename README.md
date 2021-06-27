# Math-Heuristic-Cut-Model
A math heuristic to problem cut, using the exact model.


## Para executar o projeto é necessário realizar algumas etapas.<br>
A primeira é a instalação dos pacotes necessários para a execução do projeto.<br>
Instale o conda caso não tenha em seu sistema.<br>
<br>
Antes de criar o ambiente execute o seguinte comando do conda para confirmar que o canal conda-forge esteja disponível, isso é necessário para que o conda consiga instalar todos os pacotes.<br>
`conda config --add channels conda-forge`<br>

Este projeto foi desenvolvido utilizando conda, então é possível encontrar o requirements para seu ambiente.<br>
Crie um novo ambiente com o comando abaixo, execute o seguinte comando dentro da pasta do projeto: <br>
`conda create --name math-heuristic-model --file requirements.txt`<br>

Após o ambiente ser criado e instalado as dependências, ative o env com o comando abaixo:<br>
`conda activate math-heuristic-model`<br>

Agora basta executar o projeto, para a execução utilize o seguinte comando:<br>
`python3 Cluster.py`

