# FlowerClassifierCNN

## Descrição
Este projeto é uma rede neural convolucional (CNN) desenvolvida em Python usando TensorFlow/Keras para a classificação de imagens de flores. O modelo é treinado com um dataset de flores disponível no TensorFlow e é capaz de identificar diferentes categorias de flores com base nas imagens fornecidas.

## Tecnologias Utilizadas
- Python  
- TensorFlow/Keras  
- Matplotlib  

## Estrutura do Código
1. **Carregamento do Dataset**: O código baixa e processa um conjunto de imagens de flores.
2. **Pré-processamento dos Dados**: Normaliza os valores dos pixels e divide os dados em treino e validação.
3. **Construção do Modelo**: Implementa uma CNN com múltiplas camadas convolucionais e densas.
4. **Treinamento**: O modelo é treinado por 10 épocas com otimização Adam.
5. **Visualização de Resultados**: Exibe gráficos de acurácia para análise do desempenho do modelo.

## Como Executar
1. Instale as dependências:
   ```bash
   pip install tensorflow matplotlib
   ```
2. Execute o script Python:
   ```bash
   python ml_image_classification.py
   ```

## Resultados Esperados
Após o treinamento, o modelo será capaz de classificar corretamente imagens de flores com uma boa taxa de acerto. Os gráficos de desempenho ajudam a visualizar a evolução do modelo ao longo das épocas.

## Contribuição
Sinta-se à vontade para modificar e melhorar o modelo. Sugestões e melhorias são bem-vindas!

