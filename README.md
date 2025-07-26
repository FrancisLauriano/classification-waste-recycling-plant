
<p align="center">
  <img src="https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/static/v1?label=Pandas&message=2.x&color=150458&style=for-the-badge&logo=pandas"/>
  <img src="https://img.shields.io/static/v1?label=NumPy&message=1.x&color=013243&style=for-the-badge&logo=numpy"/>
  <img src="https://img.shields.io/static/v1?label=Scikit-Learn&message=1.x&color=f7931e&style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/static/v1?label=TensorFlow&message=2.x&color=FF6F00&style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/static/v1?label=Status&message=Em%20Desenvolvimento&color=yellow&style=for-the-badge"/>
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=green&style=for-the-badge"/>
</p>

<h1 align="center">♻️ WARP - Waste Recycling Plant Classification</h1>

---

## 📑 Índice

- [📑 Índice](#-índice)
- [📝 Descrição](#-descrição)
- [🎯 Objetivo](#-objetivo)
- [🗺️ Estrutura do Projeto](#️-estrutura-do-projeto)
- [📊 Dataset](#-dataset)
  - [🔗 Onde obter os dados?](#-onde-obter-os-dados)
  - [💻 Como usar os dados?](#-como-usar-os-dados)
  - [⚠️ Observação](#️-observação)
- [🔥 Atividades do Projeto](#-atividades-do-projeto)
  - [✅ **ATIV-04-ET-01 — Análise do Dataset**](#-ativ-04-et-01--análise-do-dataset)
  - [✅ **ATIV-04-ET-02 — Pesquisa de Métodos + Propostas**](#-ativ-04-et-02--pesquisa-de-métodos--propostas)
  - [✅ **ATIV-04-ET-03 — Metodologia, Resultados e Conclusões**](#-ativ-04-et-03--metodologia-resultados-e-conclusões)
- [🧠 Pipeline do Projeto](#-pipeline-do-projeto)
- [📑 Metodologia](#-metodologia)
- [📈 Métricas de Avaliação](#-métricas-de-avaliação)
- [🚀 Como Executar o Projeto](#-como-executar-o-projeto)
  - [1️⃣ Clone este repositório:](#1️⃣-clone-este-repositório)
  - [2️⃣ Acesse a pasta do projeto:](#2️⃣-acesse-a-pasta-do-projeto)
  - [3️⃣ Crie um ambiente virtual:](#3️⃣-crie-um-ambiente-virtual)
  - [4️⃣ Ative o ambiente virtual:](#4️⃣-ative-o-ambiente-virtual)
  - [5️⃣ Instale as dependências:](#5️⃣-instale-as-dependências)
  - [6️⃣ Execute os notebooks ou scripts na pasta `/notebooks`.](#6️⃣-execute-os-notebooks-ou-scripts-na-pasta-notebooks)
- [🔧 Tecnologias e Bibliotecas](#-tecnologias-e-bibliotecas)
- [👥 Squad](#-squad)
- [📜 Licença](#-licença)

---

## 📝 Descrição

O **Warp Waste Classification** é um projeto de Machine Learning que utiliza técnicas de Visão Computacional para classificar resíduos recicláveis. O modelo é capaz de identificar materiais como **papel, plástico, vidro, metal e lixo orgânico**, auxiliando na automação de processos de triagem de resíduos sólidos.

---

## 🎯 Objetivo

- Desenvolver modelos preditivos para classificar imagens de resíduos recicláveis.  
- Contribuir para soluções de sustentabilidade e automação na gestão de resíduos.  
- Aplicar técnicas de aprendizado de máquina e visão computacional.  

---

## 🗺️ Estrutura do Projeto

```plaintext
warp-waste-classification/
├── data/              → Dados brutos (não incluídos no GitHub)
├── docs/              → Documentação e apresentações
├── notebooks/         → Notebooks de exploração, EDA, modelagem e análise
├── outputs/           → Resultados, gráficos e modelos treinados
├── src/               → Scripts Python (pré-processamento, modelagem)
├── .gitignore         → Arquivos e pastas ignorados no versionamento
├── LICENSE            → Licença MIT
├── README.md          → Documentação do projeto (este arquivo)
└── requirements.txt   → Dependências do projeto
```

---

## 📊 Dataset

- 📦 **Nome Dataset no Kaggle:** Warp Waste Recycling Plant Dataset  
- 🔗 **Link Kaggle:** [Kaggle - Warp Waste Recycling Plant Dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset)  

- 📦 **Nome da Pasta no Google Drive:** data  
- 🔗 **Link Google Drive (Compartilhado pelo Squad):** [Google Drive - Pasta /data](https://drive.google.com/drive/folders/18jEJJAZl1Fj-ejJhinqM2gYWYpPIlFpt?usp=drive_link)  

**Descrição:**  
O dataset contém imagens categorizadas de resíduos recicláveis, incluindo papel, metal, vidro, plástico, entre outros, utilizadas para treinamento, validação e teste de modelos de classificação de imagens.

⚠️ Os dados utilizados neste projeto **não estão armazenados neste repositório**, devido às limitações de espaço e às políticas do GitHub.

### 🔗 Onde obter os dados?                                                                                                                           
✅ **Opção 1 — Kaggle:**                                                                                                
Faça o download diretamente do Kaggle:                                                                                            
🔗 [Warp Waste Recycling Plant Dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset)                                             

<br>

✅ **Opção 2 — Google Drive:**                                                  
➡️ O dataset já foi baixado do Kaggle e está disponível na pasta `/data` compartilhada no Google Drive do projeto, associado à conta:                            
📧 `squadclassificacaodosresiduos@gmail.com`                                                                                           
                                   
✔️ **Importante:** Esta pasta é mantida atualizada e sincronizada pelo squad, evitando a necessidade de realizar o download manual do Kaggle.      

<br>

### 💻 Como usar os dados?                                                              
➡️ **Trabalhando localmente (VSCode ou JupyterLab):**                                                                                
1️⃣ Baixe o dataset diretamente do Kaggle.                                                                        
2️⃣ Extraia o conteúdo na pasta local do projeto chamada `/data`.                                                             
☑️ **Ou:** Copie a pasta `/data` diretamente do Google Drive compartilhado.                                                              
3️⃣ Garanta que a estrutura de diretórios seja mantida para o correto funcionamento dos notebooks e scripts.                                                     

➡️ **Utilizando Google Colab:**                                                                             
✅ O ideal é montar seu Google Drive no Colab e acessar os arquivos diretamente da pasta `/data` sincronizada no Drive.                                            
✅ Assim, não é necessário realizar upload manual dos dados sempre que iniciar uma sessão.                                                            
  - **Exemplo de montagem no Colab:**
  ```plaintext
  from google.colab import drive
  drive.mount('/content/drive')

  # Caminho até a pasta data no Google Drive
  data_path = '/content/drive/MyDrive/data'
  ```

### ⚠️ Observação
- Os arquivos da pasta `data` estão listadas no arquivo `.gitignore` para não serem enviada ao GitHub.  

---

## 🔥 Atividades do Projeto

### ✅ **ATIV-04-ET-01 — Análise do Dataset**
- ✔️ Verificar integridade dos arquivos.
- ✔️ Checar consistência dos metadados.
- ✔️ Avaliar qualidade das imagens.
- ✔️ Analisar distribuição das classes.
- ✔️ Identificar e tratar imagens duplicadas.
  
> 📄 **Notebook:**  
> Análise do Dataset encontra-se em [`./notebooks/Etapa-1_WaRP_Dataset_Analysis.ipynb`](./notebooks/Etapa-1_WaRP_Dataset_Analysis.ipynb).
> 
> A apresentação da Análise do Dataset encontra-se em [`./docs/Etapa_01_Apresetacao.pdf`](./docs/Etapa_01_Apresetacao.pdf).

<br>

### ✅ **ATIV-04-ET-02 — Pesquisa de Métodos + Propostas**
- ✔️ Implementar e testar métodos da literatura (disponíveis no próprio Kaggle e outros artigos).
- ✔️ Propor melhorias, como:
  - Otimização de hiperparâmetros.
  - Arquiteturas mais leves e rápidas.
  - Redução do conjunto de treino mantendo a performance.

> 📄 **Notebook:**  
> Análise de métodos da literatura e propor ideias para métodos próprios encontra-se em [`./docs/Etapa_02_metodos_da_literatura_e_ideias_para_metodos_proprios.pdf`](./docs/Etapa_02_metodos_da_literatura_e_ideias_para_metodos_proprios.pdf).

<br>

### ✅ **ATIV-04-ET-03 — Metodologia, Resultados e Conclusões**
- ✔️ Documentar a metodologia adotada.
- ✔️ Comparar resultados entre modelos da literatura e modelos próprios.
- ✔️ Apresentar tabelas, gráficos e métricas.
- ✔️ Elaborar conclusões, aprendizados e sugestões de melhorias futuras.

> 📄 **Notebook:**  
> O Treinamento encontra-se em [`./notebooks/Etapa-3_ResNet50_WaRP_KFold_Augmentation_Undersampling_Class_weight_V2.ipynb`](./notebooks/Etapa-3_ResNet50_WaRP_KFold_Augmentation_Undersampling_Class_weight_V2.ipynb).

<br>

> 📄 **Nota:**  
> O roteiro do Projeto do Bootcamp encontra-se em [`./docs/projeto-atividade-04.pdf`](./docs/projeto-atividade-04.pdf).

---

## 🧠 Pipeline do Projeto

```mermaid
graph TD
A[Análise do Dataset] --> B[Pré-processamento]
B --> C[Modelagem com métodos da literatura]
C --> D[Testes com melhorias e modelos próprios]
D --> E[Análise de Resultados]
E --> F[Conclusões e Entregas]
```

---

## 📑 Metodologia

- ✅ **Análise Exploratória dos Dados (EDA)**
- ✅ **Pré-processamento**
- ✅ **Treinamento e Avaliação de Modelos**
- ✅ **Aprimoramento de Modelos (Hiperparâmetros, Arquitetura)**
- ✅ **Análise de Resultados e Conclusões**

---

## 📈 Métricas de Avaliação

- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clone este repositório:

```bash
git clone https://github.com/FrancisLauriano/classification-waste-recycling-plant.git
```

### 2️⃣ Acesse a pasta do projeto:

```bash
cd seu-repositorio
```

### 3️⃣ Crie um ambiente virtual:

```bash
python -m venv venv
```

### 4️⃣ Ative o ambiente virtual:

- Windows:

```bash
venv\Scripts\activate
```

- Linux/Mac:

```bash
source venv/bin/activate
```

### 5️⃣ Instale as dependências:

```bash
pip install -r requirements.txt
```

### 6️⃣ Execute os notebooks ou scripts na pasta `/notebooks`.

---

## 🔧 Tecnologias e Bibliotecas

- **Linguagem:** Python 3.10+
- **Bibliotecas:**
  - Pandas
  - NumPy
  - Scikit-Learn
  - TensorFlow 
  - OpenCV
  - Matplotlib
  - Seaborn
  - Jupyter Notebook
- **Ferramentas:**
  - Kaggle
  - Google Colab ou JupyterLab
  - Git e GitHub

---

## 👥 Squad

<table align="center">
  <tr>
    <td align="center">
      <a href="https://www.linkedin.com/in/ana-sofia-moura-27b003248/">
        <img src="https://avatars.githubusercontent.com/u/109629293?v=4" width="115px;" alt="Ana Sofia Profile Image"/><br>
      </a>
      <sub>
          <b>Ana Sofia Moura</b>
         </sub>
    <br>
        <a href="https://github.com/Sun-cs-Sol">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/ana-sofia-moura-27b003248/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://github.com/Elineison">
        <img src="https://avatars.githubusercontent.com/u/153145464?v=4" width="115px;" alt="Elineison de Sousa Profile Image"/><br>
      </a>
      <sub>
          <b>Elineison de Sousa</b>
         </sub>
    <br>
        <a href="https://github.com/Elineison">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/elineison-inacio-de-sousa/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/felipemiguel99/">
        <img src="https://avatars.githubusercontent.com/u/140664219?v=4" width="115px;" alt="Felipe Miguel Profile Image"/><br>
      </a>
      <sub>
          <b>Felipe Miguel</b>
         </sub>
    <br>
        <a href="https://github.com/fpmiguels">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/felipemiguel99/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/francis-lauriano-9b674a27b/">
        <img src="https://avatars.githubusercontent.com/u/130801505?v=4" width="115px;" alt="Francis Lauriano Profile Image"/><br>
      </a>
      <sub>
          <b>Francis Lauriano</b>
         </sub>
    <br>
        <a href="https://github.com/FrancisLauriano">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/francis-lauriano-9b674a27b/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/iza-francine/">
        <img src="https://avatars.githubusercontent.com/u/212177943?v=4" width="115px;" alt="Iza Vieira Profile Image"/><br>
      </a>
      <sub>
          <b>Iza Vieira</b>
         </sub>
    <br>
        <a href="https://github.com/iza1603francine">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/iza-francine/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/madelu-lopes-089388120/">
        <img src="https://avatars.githubusercontent.com/u/176135579?v=4" width="115px;" alt="Madelu Lopes Profile Image"/><br>
      </a>
      <sub>
          <b>Madelu Lopes</b>
         </sub>
    <br>
        <a href="https://github.com/Mads8760">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/madelu-lopes-089388120/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/mariana-angeli/">
        <img src="https://avatars.githubusercontent.com/u/172328698?v=4" width="115px;" alt="Mariana Angeli Profile Image"/><br>
      </a>
      <sub>
          <b>Mariana Angeli</b>
         </sub>
    <br>
        <a href="https://github.com/marianaangeli">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/mariana-angeli/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    <td align="center">
      <a href="https://www.linkedin.com/in/rodrigo-rocha-19249170/">
        <img src="https://avatars.githubusercontent.com/u/126805602?v=4" width="115px;" alt="Rodrigo Rocha Profile Image"/><br>
      </a>
      <sub>
          <b>Rodrigo Rocha</b>
         </sub>
    <br>
        <a href="https://github.com/Rodrigoldarocha">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <br>
        <a href="https://www.linkedin.com/in/rodrigo-rocha-19249170/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </td>
    
  </tr>
</table>

---

## 📜 Licença

Este projeto está sob a licença **MIT** — Consulte o arquivo [LICENSE](LICENSE) para mais informações.  

Projeto desenvolvido exclusivamente para fins educacionais no contexto do **Bootcamp em Machine Learning**.

---