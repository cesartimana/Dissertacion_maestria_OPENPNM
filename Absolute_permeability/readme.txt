Os arquivos rodam com OpenPNM 3.2.0

O arquivo "Berea_max_d.pnm" tem a informação da rede de poros para Berea, incluido G, beta. pore.diameter.

A peremabilidade relativa k_r no caso de drenagem primaria para um sistema oleo-agua é obtida usando 2 métodos:

1) Método do OpenPNM

    -O arquivo usado para simular é "Usando_OpenPNM.py".
    -A metodologia é a do tutorial do site de OpenPNM. "spheres_and_cylinders": https://openpnm.org/examples/applications/relative_permeability.html
    -Assume-se que a rede esta composta por esferas e cilindros, mas o modelo "op.models.collections.geometry.spheres_and_cylinders" do OpenPNM não foi usado.
    -Os modelos "op.models.collections.geometry" recalculam propriedades como "diameter" e "volume",colocando dados randomicos. Por tanto:
        -pore.diameter é calculado usando pore.volume
        -Para throat.diameter, escolha-se os dados de throat.inscribed_diameter da rede. Logo, se Dt > Dp(vizinhos), se reduz o valor de Dt com solve_throat_diameter
        -Os comprimentos / lenght para os elementos (poro/garganta) de cada conduite é calculado usando _cf.conduit_lenght_tubes
        -throat.hydraulic_size_factors é calculado manualmente, usando como referencia o codigo op.models.geometry.conduit_length.spheres_and_cylinders
        -solve_throat_diameter e _cf.conduit_lenght_tubes são funções criadas.
    -Os resultados de saturação e k_{r,oleo} são salvados nos arquivos "info_OpenPNM".



2) Método novo proposto

    -O arquivo usado para simular é "Teste_rede.py".
    -Esse arquivo usa outros 3 arquivos:
        - _algorithm_class.py: Codigo para a clase "Primary_Drainage"
        - _conductance_funcs.py: Tem funções relacionadas ao calculo da conductancia
        - _invasion_funcs.py: Tem funções relacaionadas a estimativa de invasão e obtenção de cluster por phase e elemento
    -Os resultados de saturação e k_r são salvados nos arquivos "info_rel_perm...". "0" para agua, "1" para oleo.

O gráfico que compara os resultados de ambos os métodos é obtido ao rodar "Plotting_data.py".
Também são plotados os dados experimentais da mostra da Berea (Oak 1990: Three-Phase Relative Permeability of Water-Wet Berea).
Esses dados estão salvados nos arquivos "Dados_..._Berea.dat"

Legenda do grafico:
-exp.: Dados experimentais
-OpenPNM: Metodo 1
-sim.: Método 2
