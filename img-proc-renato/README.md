# Processamento de imagem de satélite com redes neurais

### Problema a ser resolvido

Dada uma stream de imagens aéreas/espaciais contíguas e sequenciais sobre uma região, retorne um mapa dos seguintes possíveis acidentes ambientais (pode não haver nenhum acidente): detecção de óleo na água, deslizamento e rachaduras na terra.

### Soluções

Para cada imagem fazer um processo de predição por redes neurais que funciona assim:

A partir da matriz gerada, fazer um mapeamento da matriz para uma outra matriz maior de mapa, e, a partir das imagens de input, compor numa camada abaixo desse mapa outro mapa com a média das imagens reais para poder fazer um overlay e mostrar o acidente em tempo real.

Google earth permite coleta de dados como esses. Aqui estão alguns exemplos extraídos:


Essas imagens foram coletadas pegando uma foto antiga e outra atual sobre o Mato Grosso e comparando a diferença (Structual Similarity) entre elas.

Como é possível ver, quanto mais branco, mais desmatamento. A ideia seria montar um dataset com vários exemplos desses treinar a rede para receber a imagem de ‘depois’ e gerar a imagem de ‘diferença’, dessa forma podemos plotar num mapa. Após o treinamento do modelo, podemos expandir as categorias para as categorias explicitadas no problema, haja visto que existe a necessidade de dados, algo que ainda não possuímos. Com o modelo será possível expandir para dados novos, ou seja, o modelo para desmatamento permite a criação de outros modelos para outros desastres vistos do céu apenas com mais dados.

O modelo em questão seria uma Generative Neural Network (GAN) [1]. Este modelo faz um mapeamento de imagens, transformando uma imagem orbital de um terreno em uma imagem preto e branco representando desflorestamento, como explicado anteriormente. Ele é composto por duas redes neurais durante seu treinamento: a geradora e a discriminadora. A geradora gera imagens novas a partir do input e a discriminadora é uma rede treinada para dizer se uma imagem é real ou não. 

O treinamento consiste da geradora produzir uma imagem a partir do input e a discriminadora recebe, sem saber qual é qual, a imagem produzida pela rede e a imagem correta. A discriminadora é então “punida” ou “presenteada” por errar ou acertar, respectivamente, e, dependendo do seu resultado, o inverso acontece com a geradora: se a discriminadora erra, então a geradora é “presenteada” e vice versa. Existem diversos artigos científicos sobre esse modelo, bem como exemplos desse tipo de rede em todos os grandes frameworks de inteligência artificial.

Abaixo nós temos alguns exemplo de uma versão extremamente simples e pouco treinada desse modelo, onde a imagem da esquerda é o input, a do meio é a imagem que a rede neural gerou e a da direita, o resultado certo:


Essa rede, porém, ainda necessita de muito treinamento e trabalho. Acompanhe o que acontece com uma imagem nova:


Portanto essa solução me parece viável mas não está completa e o processo de melhorar esse modelo pode vir a consumir muito tempo. 

A partir de agora, vamos supor que exista 1) um modelo funcional e 2) os dados. 

Como resolver o problema por inteiro? É sabido que redes neurais, depois de treinadas retornam resultados rápidos, portanto isso não seria um problema.

### Referências

- [1] Paper sobre rede neural GAN que permite tal transformações
https://arxiv.org/pdf/1611.07004.pdf

- Diferença estrutural entre imagens
https://en.wikipedia.org/wiki/Structural_similarity

- Projeto semelhante (voltado apenas para desflorestamento)
https://towardsdatascience.com/land-use-and-deforestation-in-the-brazilian-amazon-5467e88933b

