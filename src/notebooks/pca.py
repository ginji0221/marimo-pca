import marimo

__generated_with = "0.15.2"
app = marimo.App(
    width="medium",
    app_title="主成分分析",
    css_file="../../css/mininini.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import StandardScaler
    return StandardScaler, go, make_subplots, mo, np, pd, px


@app.cell
def _(mo):
    # ページタイトル
    page_title = mo.md("# 主成分分析 (PCA)")
    # page_title
    return (page_title,)


@app.cell
def _(StandardScaler, mo, np, pd):
    # データセットの読み込み
    df_iris = pd.read_csv("datasets/iris.csv")

    data = df_iris.drop(columns="Species")
    target = df_iris["Species"]
    dimensions = data.columns
    unique_target = np.unique(target)

    # 標準化オブジェクト
    scaler = StandardScaler()

    # スイッチ作成
    switches_dimension = mo.ui.array(
        [mo.ui.switch(value=True, label=_dimension) for _dimension in dimensions],
        label="demensions"
    )
    switches_target = mo.ui.array(
        [mo.ui.switch(value=True, label=_target) for _target in unique_target],
        label="target"
    )
    switch_standardize = mo.ui.switch(value=True, label="Standardization")
    switch_show_constraint_01 = mo.ui.switch(value=True, label="Constraint 01")
    switch_show_constraint_02 = mo.ui.switch(value=True, label="Constraint 02")
    switch_show_vector_a = mo.ui.switch(value=True, label="vector a")
    switch_show_vector_b = mo.ui.switch(value=True, label="vector b")

    # ドロップダウン作成
    dropdown_2d_dimension_01 = mo.ui.dropdown(
        options={_dimension : _i for _i, _dimension in enumerate(dimensions)},
        value="SepalLength",
        label="Dim 01 : ",
    )
    dropdown_2d_dimension_02 = mo.ui.dropdown(
        options={_dimension : _i for _i, _dimension in enumerate(dimensions)},
        value="SepalWidth",
        label="Dim 02 : ",
    )

    # スライダーの作成
    slider_corr = mo.ui.slider(
        start=-1, 
        stop=1, 
        step=0.1, 
        label="Corr", 
        value=0.5,
        include_input=True
    )

    # state作成
    get_dropdown_2d_dim01_state, set_dropdown_2d_dim01_state = mo.state(0)
    get_dropdown_2d_dim02_state, set_dropdown_2d_dim02_state = mo.state(1)
    return (
        data,
        df_iris,
        dimensions,
        dropdown_2d_dimension_01,
        dropdown_2d_dimension_02,
        get_dropdown_2d_dim01_state,
        get_dropdown_2d_dim02_state,
        scaler,
        set_dropdown_2d_dim01_state,
        set_dropdown_2d_dim02_state,
        slider_corr,
        switch_show_constraint_01,
        switch_show_constraint_02,
        switch_show_vector_a,
        switch_show_vector_b,
        switch_standardize,
        switches_dimension,
        switches_target,
        target,
        unique_target,
    )


@app.cell
def _(
    data,
    df_iris,
    dimensions,
    mo,
    px,
    switches_dimension,
    switches_target,
    target,
    unique_target,
):
    # ===== 1章 =====

    # スイッチによるディメンション, ターゲットの選択
    _dimensions_selected = dimensions[switches_dimension.value]
    _target_selected = unique_target[switches_target.value]

    _data = data[_dimensions_selected][target.isin(_target_selected)]
    _target = target[target.isin(_target_selected)]

    # 散布図の作成
    _fig = px.scatter_matrix(
        _data,
        dimensions=_dimensions_selected,
        color=_target,
        template="plotly_dark",
        opacity=0.5,
        # width=700,
        # height=600
    )

    _fig.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig.data]
    for _i, _trace in enumerate(_fig.data):
        # _trace.marker.color = "rgba(0,0,0,0)"
        _trace.marker.line.color = "black"
        _trace.marker.line.width = 1
        _trace.marker.size = 4.5

    # コンポーネント作成
    _subtitle = mo.md(
        """
        ## 1. 使用データセット

        今回は iris.csv を使用する。このデータセットの次元数は4次元であり、各特長は下記の詳細に記載している。  
        また、合計サンプル数は150件であり、品種は３種類 (Setosa, Versicolor, Virsinica) が用意されている (各品種ごとに50件ずつ)。
        """
    )

    _table = mo.vstack(
        [
            mo.md("### データフレーム"),
            df_iris,
        ],
        gap=1,
    )

    _table_description = mo.hstack(
        [
            _table,
            mo.md(
                """
                ### 詳細

                植物のアヤメの各品種別の特徴をまとめたデータ

                特徴の詳細は下記の通り

                - SepalLength  : がく片の長さ
                - SepalWidth   : がく片の幅
                - PetalLength  : 花弁の長さ
                - PetalLength  : 花弁の幅
                - Species      : アヤメの品種
                """
            )
        ],
        gap=3,
        widths=[2, 1],
    )

    _switches = mo.vstack(
        [
            switches_dimension,
            switches_target
        ],
        gap=1
    )

    _scatter_matrix_with_switch = mo.vstack(
        [
            mo.md("### 散布図 一覧"),
            mo.hstack(
                [
                    mo.ui.plotly(_fig),
                    _switches,
                ],
                gap=3,
                widths=[2, 1]
            )
        ],
        gap=1
    )

    iris_dataset_component = mo.vstack(
        [
            _subtitle,
            _table_description,
            _scatter_matrix_with_switch,
        ],
        gap=2,
    )

    # iris_dataset_component
    return (iris_dataset_component,)


@app.cell
def _(
    data,
    dimensions,
    dropdown_2d_dimension_01,
    dropdown_2d_dimension_02,
    get_dropdown_2d_dim01_state,
    get_dropdown_2d_dim02_state,
    mo,
    pd,
    px,
    scaler,
    set_dropdown_2d_dim01_state,
    set_dropdown_2d_dim02_state,
    switch_standardize,
    target,
):
    # ===== 2章 =====

    # ドロップダウンによるディメンション, ターゲットの選択
    if dropdown_2d_dimension_01.value != dropdown_2d_dimension_02.value:
        # stateの設定
        set_dropdown_2d_dim01_state(dropdown_2d_dimension_01.value)
        set_dropdown_2d_dim02_state(dropdown_2d_dimension_02.value)

        _dimensions_selected = dimensions[
            [
                dropdown_2d_dimension_01.value,
                dropdown_2d_dimension_02.value
            ]
        ]
    else:
        _dimensions_selected = dimensions[
            [
                get_dropdown_2d_dim01_state(),
                get_dropdown_2d_dim02_state()
            ]
        ]

    _data = data[_dimensions_selected]
    _target = target

    # データの標準化
    if switch_standardize.value:
        _data = pd.DataFrame(
            data=scaler.fit_transform(_data),
            columns=_dimensions_selected,
        )

    # グローバル変数追加
    data_2d = _data
    target_2d = _target

    # グラフ作成
    _fig = px.scatter(
        data_2d,
        x=dropdown_2d_dimension_01.selected_key,
        y=dropdown_2d_dimension_02.selected_key,
        color=target_2d,
        template="plotly_dark",
        opacity=0.5,
    )

    _fig.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig.data]
    for _i, _trace in enumerate(_fig.data):
        _trace.marker.line.color = "black"
        _trace.marker.line.width = 3
        _trace.marker.size = 12

    # コンポーネント作成
    _subtitle = mo.md(
        f"""
        ## 2. 標準化

        標準化は入力値の平均を0、分散（標準偏差）を1に変換し、特徴同士のスケールをそろえる処理です。  
        この処理によって、異なる単位やスケールを持つ特徴量が公平に扱われます。

        標準化の計算は以下の通りです：  

        $$
        u_1 = \\frac{{x_1 - \\bar{{x}}_1}}{{s_1}},\\quad
        u_2 = \\frac{{x_2 - \\bar{{x}}_2}}{{s_2}}
        $$

        $x_1$ : {_dimensions_selected[0]}  
        $x_2$ : {_dimensions_selected[1]}  
        $u$ : $x$ を標準化した値  
        $\\bar{{x}}$ : $x$ の平均  
        $s$ : $x$ の標準偏差
        """
    )

    _params = mo.vstack(
        [
            dropdown_2d_dimension_01, 
            dropdown_2d_dimension_02, 
            switch_standardize
        ]
    )

    _data_fig_params = mo.hstack(
        [
            mo.vstack([data_2d, _fig], gap=1),
            _params,
        ],
        gap=3,
        widths=[2, 1]
    )

    standardize_component = mo.vstack(
        [
            _subtitle,
            mo.vstack([mo.md("### 標準化の可視化"), _data_fig_params]),
        ],
        gap=3
    )

    # standardize_component
    return data_2d, standardize_component


@app.cell
def _(
    go,
    mo,
    np,
    slider_corr,
    switch_show_constraint_01,
    switch_show_constraint_02,
    switch_show_vector_a,
    switch_show_vector_b,
):
    # ===== 3章 =====

    # グローバル変数の追加
    _r = slider_corr.value # 相関係数
    _R = np.array([[1, _r], [_r, 1]]) # 相関係数行列
    _eigen_val, _eigen_vec = np.linalg.eig(_R) # 固有値, 固有ベクトル

    # グラフ作成
    # --- 主成分スコアの分散の関数 ---
    _a1 = np.linspace(-2, 2, 200)
    _a2 = np.linspace(-2, 2, 200)
    _a1, _a2 = np.meshgrid(_a1, _a2)
    _Vz1 = _a1 ** 2 + _a2 ** 2 + 2 * _r * _a1 * _a2


    _a1_vec, _a2_vec = _eigen_vec.T[0]
    _b1_vec, _b2_vec = _eigen_vec.T[1]

    _fig = go.Figure(
        data=[
            go.Surface(
                z=_Vz1, x=_a1, y=_a2,
                opacity=0.5,                     
                showscale=False,                # カラーバー非表示
                colorscale="Viridis",
                name="V<sub>z<sub>1</sub></sub>関数"
            )
        ]
    )

    _fig.update_traces(
        contours_z=dict(
            show=True, 
            usecolormap=True,
            highlightcolor="limegreen", 
            project_z=True
        ),
    )

    # --- 制約条件 01 ---
    if switch_show_constraint_01.value is True:
        _theta = np.linspace(0, 2 * np.pi, 200)
        _circle_x = np.cos(_theta)  # a1
        _circle_y = np.sin(_theta)  # a2
        _circle_z = np.zeros_like(_theta)  # z=0 plane

        _fig.add_trace(
            go.Scatter3d(
                x=_circle_x,
                y=_circle_y,
                z=_circle_z,
                mode="lines",
                line=dict(color="red", width=4),
                name="a<sub>1</sub>² + a<sub>2</sub>² = 1"
            )
        )

    # --- 制約条件 02 ---
    if switch_show_constraint_02.value is True:
        _line_x = np.linspace(-2, 2, 200)
        _line_y = - (_a1_vec / _a2_vec) * _line_x if _a2_vec != 0 else np.zeros_like(_line_x)
        _line_z = np.zeros_like(_line_x)

        _fig.add_trace(
            go.Scatter3d(
                x=_line_x,
                y=_line_y,
                z=_line_z,
                mode="lines",
                line=dict(color="blue", width=4),
                name="a<sub>1</sub> * b<sub>1</sub> + a<sub>2</sub> * b<sub>2</sub> = 0"
            )
        )

    # --- 主成分ベクトル 01 ---
    if switch_show_vector_a.value is True:
        _fig.add_traces(
            [
                go.Scatter3d(
                    x=[0, _a1_vec],
                    y=[0, _a2_vec],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="indianred", width=6),
                    showlegend=False,
                ),
                go.Cone(
                    x=[_a1_vec], y=[_a2_vec], z=[0], 
                    u=[_a1_vec], v=[_a2_vec], w=[0],
                    colorscale="reds",
                    showscale=False,
                    showlegend=True,
                    name = "vector a"
                )
            ]
        )

    # --- 主成分ベクトル 02 ---
    if switch_show_vector_b.value is True:
        _fig.add_traces(
            [
                go.Scatter3d(
                    x=[0, _b1_vec],
                    y=[0, _b2_vec],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="deepskyblue", width=6),
                    showlegend=False,
                ),
                go.Cone(
                    x=[_b1_vec], y=[_b2_vec], z=[0], 
                    u=[_b1_vec], v=[_b2_vec], w=[0],
                    colorscale="blues",
                    showscale=False,
                    showlegend=True,
                    name = "vector b"
                )
            ]
        )

    # --- レイアウト調整 ---
    _fig.update_layout(
        title="V<sub>z<sub>1</sub></sub> = a<sub>1</sub>² + a<sub>2</sub>² + 2 r<sub>xy</sub> a<sub>1</sub> a<sub>2</sub>",
        scene=dict(
            xaxis_title="a<sub>1</sub>",
            yaxis_title="a<sub>2</sub>",
            zaxis_title="V<sub>z<sub>1</sub></sub>",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        template="plotly_dark",
        showlegend=True,
    )

    # コンポーネント作成
    _params = mo.vstack(
        [
            slider_corr,
            switch_show_constraint_01,
            switch_show_constraint_02,
            switch_show_vector_a,
            switch_show_vector_b,
        ],
        gap=1
    )

    _fig_params = mo.hstack(
        [
            mo.ui.plotly(_fig),
            _params,
        ],
        gap=3,
        widths=[2, 1]
    )

    var_func_component = mo.vstack(
        [
            mo.md("### 主成分スコア$z_1$の分散関数"),
            _fig_params,
        ]
    )

    # var_func_component
    return (var_func_component,)


@app.cell
def _(data_2d, make_subplots, mo, np, pd, px, target, var_func_component):
    # 主成分スコアの計算
    _R = data_2d.corr().values
    eigen_val, eigen_vec = np.linalg.eig(_R)

    _data_transformed_dim_02 = np.dot(data_2d.values, eigen_vec)
    _data_transformed_dim_01 = _data_transformed_dim_02[:, [0]]
    _zeros_col = np.zeros((_data_transformed_dim_02.shape[0], 1))
    _data_transformed_dim_01 = np.hstack(
        (
            _zeros_col,
            _data_transformed_dim_01,
        )
    )

    _df_dim_01 = pd.DataFrame( # 1次元用
        data=_data_transformed_dim_01,
        columns=["z0", "z1"]
    )
    _df_dim_02 = pd.DataFrame( # 2次元用
        data=_data_transformed_dim_02,
        columns=["z1", "z2"]
    )

    _df_dim_01 = pd.concat([_df_dim_01, target], axis=1)
    _df_dim_02 = pd.concat([_df_dim_02, target], axis=1)

    # グラフ作成
    _fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("PCA with 1 Dimension", "PCA with 2 Dimension")
    )

    _fig_dim_01 = px.scatter(
        _df_dim_01,
        x="z1", y="z0",
        color="Species",
        title="PCA with 1 Dimension",
        opacity=0.5,

    )
    _fig_dim_02 = px.scatter(
        _df_dim_02,
        x="z1", y="z2",
        color="Species",
        title="PCA with 2 Dimension",
        opacity=0.5,
    )

    for _trace_dim_01 in _fig_dim_01.data:
        _trace_dim_01.showlegend = False
        _fig.add_trace(
            _trace_dim_01,
            row=1, col=1
        )
    for _trace_dim_02 in _fig_dim_02.data:
        _fig.add_trace(
            _trace_dim_02,
            row=1, col=2
        )

    _fig.update_xaxes(title_text="z<sub>1</sub>", row=1, col=1)
    _fig.update_xaxes(title_text="z<sub>1</sub>", row=1, col=2)
    _fig.update_yaxes(title_text="z<sub>2</sub>", row=1, col=2)

    _fig.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig.data]
    for _i, _trace in enumerate(_fig.data):
        _trace.marker.line.color = "black"
        _trace.marker.line.width = 3
        _trace.marker.size = 12

    # コンポーネント作成
    _subtitle = mo.md(
        """
        ## 3. 主成分の分散最大化

        主成分分析の目的は、入力データに対して分散が最大となる新たな軸を探すことにある。  
        また、この軸は入力データの次元数と同じ数だけ存在し、この軸に射影した入力データの値を主成分という。
        """
    )

    _img = mo.image(src="img/pca_image_01.png", width="960px", height="540px")

    _description_01 = mo.md(
        """
        この時、分散が最も大きくなる軸における主成分を第1主成分といい、下記で求まる。

        $$
        z_1 = a_1 u_1 + a_2 u_2
        $$

        $z1$ : 第1主成分  
        $\\mathbf{{a}} = \\begin{{pmatrix}} a_1 \\quad a_1 \\end{{pmatrix}}$ : 主成分ベクトル


        また、主成分スコア$z_1$における分散$V_{{z}_{{1}}}$は下記で求められる。  
        $r_{{x_{1} x_{2}}}$は$x_1$, $x_2$の相関係数である。


        $$
        V_{{z}_{{1}}} = a_1^2 + a_2^2 + 2 r_{{x_{1} x_{2}}} a_1 a_2
        $$

        $V_{{z}_{{1}}}$を最大化する$\\mathbf{{a}}$は下記の固有値、固有ベクトルの問題に帰着する。  
        $R$は相関係数行列であり、$\\lambda$は固有値 (分散$V_{{z}_{{1}}}$と等価) である。

        $$
        R \\mathbf{{a}} = \\lambda \\mathbf{{a}}
        $$
        """
    )

    _df_fig = mo.vstack(
        [
            mo.md("### 主成分スコア"),
            _df_dim_02, 
            mo.md("### 主成分スコアの散布図"),
            _fig,
        ],
        gap=1
    )

    _description_02 = mo.md(
        """
        今回はiris datasetから２つの特徴を選択して、主成分スコアを計算した。  
        下記に計算結果と散布図を示す。
        """
    )

    maximize_var_component = mo.vstack(
        [
            _subtitle,
            _img,
            _description_01,
            var_func_component,
            _description_02,
            _df_fig,
        ],
        gap=3,
    )

    # maximize_var_component
    return eigen_val, eigen_vec, maximize_var_component


@app.cell
def _(eigen_val, mo, np, pd, px):
    # ===== 4章 =====

    _df_evr = pd.DataFrame({
        "Principal Component": ["PC1", "PC2"],
        "Explained Variance Ratio": [
            eigen_val[0] / np.sum(eigen_val), 
            eigen_val[1] / np.sum(eigen_val)
        ]
    })

    _fig = px.bar(
        _df_evr,
        x="Principal Component",
        y="Explained Variance Ratio",
        title="Explained Variance Ratio by Principal Component",
    )

    _fig.update_layout(
        bargap=0.5,         # 棒と棒の間の間隔（0〜1）
        bargroupgap=0.0      # グループ間の間隔（0で密着）
    )

    # コンポーネント作成
    _subtitle = mo.md(
        """
        ## 4. 寄与率

        寄与率は各主成分が元の情報をどの程度説明できるか定量的に表したものである。  
        寄与率は下記の計算式で求められる。

        $$
        EVR_{{1}} = \\frac{{\\lambda_{{1}}}}{{\\lambda_{{1}} + \\lambda_{{2}}}} \\ , \\quad
        EVR_{{2}} = \\frac{{\\lambda_{{2}}}}{{\\lambda_{{1}} + \\lambda_{{2}}}}
        $$

        $EVR_{{n}}$ : 第n主成分の寄与率  
        $\\lambda_{{n}}$ : 第n主成分ベクトル (固有ベクトル) に対応する固有値  
        """
    )

    _df_fig = mo.vstack(
        [
            mo.md("### 寄与率の可視化"),
            mo.hstack([_df_evr, _fig], gap=1, widths=[1, 2]),
        ],
        gap=1
    )

    evr_component = mo.vstack(
        [_subtitle, _df_fig],
        gap=3
    )

    # evr_component
    return (evr_component,)


@app.cell
def _(eigen_val, eigen_vec, mo, np, pd, px):
    # ===== 5章 =====

    # 因子負荷量の計算
    _r_z1x1 = np.sqrt(eigen_val[0]) * eigen_vec.T[0][0]
    _r_z1x2 = np.sqrt(eigen_val[0]) * eigen_vec.T[0][1]
    _r_z2x1 = np.sqrt(eigen_val[1]) * eigen_vec.T[1][0]
    _r_z2x2 = np.sqrt(eigen_val[1]) * eigen_vec.T[1][1]

    # グラフの作成
    _df_factor_loadings = pd.DataFrame({
        "Principal Component": ["PC1", "PC1", "PC2", "PC2"],
        "Feature": ["x1", "x2", "x1", "x2"],
        "Factor Loading": [_r_z1x1, _r_z1x2, _r_z2x1, _r_z2x2]
    })

    # 棒グラフ（色で主成分を分け、特徴量ごとに並べる）
    _fig = px.bar(
        _df_factor_loadings,
        x="Principal Component",
        y="Factor Loading",
        color="Feature",
        barmode="group",     # 横に並べる
        height=400,
        title="Factor Loadings by Feature and Principal Component"
    )


    _fig.update_layout(
        bargap=0.5,         # 棒と棒の間の間隔（0〜1）
        bargroupgap=0.0      # グループ間の間隔（0で密着）
    )

    # コンポーネント作成
    _subtitle = mo.md(
        """
        ## 5. 因子負荷量

        因子負荷量は、各主成分（PC）が元の特徴量（特徴変数）をどれだけ強く反映しているかを示す指標である。  
        これは、**主成分と元の変数との相関係数**として定義され、次の式で計算される。

        $$
        r_{{z_{{1}} x_{{1}}}} = \\sqrt{{\\lambda_1}} a_1 \\, \\quad
        r_{{z_{{1}} x_{{2}}}} = \\sqrt{{\\lambda_1}} a_2 \\, \\quad
        r_{{z_{{2}} x_{{1}}}} = \\sqrt{{\\lambda_2}} b_1 \\, \\quad
        r_{{z_{{2}} x_{{2}}}} = \\sqrt{{\\lambda_2}} b_2
        $$

        $r_{{z_{{n}} x_{{m}}}}$ : 第n主成分$z_{{n}}$と第m特徴量$x_{{m}}$の相関係数
        """
    )

    _df_fig = mo.vstack(
        [
            mo.md("### 因子負荷量の可視化"),
            mo.hstack([_df_factor_loadings, _fig], gap=1, widths=[1, 2]),
        ],
        gap=1
    )

    factor_loading_component = mo.vstack(
        [_subtitle, _df_fig],
        gap=3
    )

    # factor_loading_component
    return (factor_loading_component,)


@app.cell
def _(data, dimensions, mo, np, pd, px, scaler, target):
    # ===== 6章 =====

    # 使用するデータ (4次元すべて使用)
    _data = data
    _target = target
    _dimensions = dimensions

    # 標準化
    _data_scale = pd.DataFrame(
        data=scaler.fit_transform(_data),
        columns=_dimensions,
    )

    # 相関係数行列
    _df_R = _data_scale.corr()
    _R = _df_R.values

    # 固有値、固有ベクトル
    _eigen_val, _eigen_vec = np.linalg.eig(_R)

    # 主成分スコア
    _data_transformed = np.dot(_data_scale.values, _eigen_vec)
    _df_data_transformed = pd.DataFrame(
        data=_data_transformed,
        columns=[f"z<sub>{_i+1}</sub>" for _i in range(_data_transformed.shape[1])]
    )

    # 寄与率の計算
    _evr = _eigen_val / np.sum(_eigen_val)
    _df_evr = pd.DataFrame(
        {
            "Principal Component": [f"PC{_i+1}" for _i in range(len(_evr))],
            "Explained Variance Ratio": _evr,
        }
    )

    # 因子負荷量の計算
    _factor_loading_matrix = np.dot(
        np.diag(np.sqrt(_eigen_val)),
        _eigen_vec.T
    )

    _dict_factor_loading = {
        "Principal Component": [],
        "Feature": [],
        "Factor Loading": []
    }
    for _z_idx in range(len(_eigen_val)):
        for _x_idx in range(len(_dimensions)):
            _dict_factor_loading["Principal Component"].append(f"PC{_z_idx+1}")
            _dict_factor_loading["Feature"].append(f"x{_x_idx+1}")
            _dict_factor_loading["Factor Loading"].append(_factor_loading_matrix[_z_idx][_x_idx])
    _df_factor_loading = pd.DataFrame(_dict_factor_loading)

    # グラフ化

    # データセットの可視化
    _fig_dataset = px.scatter_matrix(
        _data_scale,
        dimensions=_dimensions,
        color=_target,
        template="plotly_dark",
        opacity=0.5,
    )

    _fig_dataset.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig_dataset.data]
    for _i, _trace in enumerate(_fig_dataset.data):
        _trace.marker.line.color = "black"
        _trace.marker.line.width = 1
        _trace.marker.size = 4.5

    _dataset_component = mo.vstack(
        [
            mo.md("### データセット (標準化済み)"),
            mo.hstack(
                [_data_scale, _fig_dataset],
                gap=1,
                widths=[1, 2]
            )
        ],
        gap=1,
    )

    # 主成分スコアの可視化
    _fig_data_transformed = px.scatter_matrix(
        _df_data_transformed,
        dimensions=[f"z<sub>{_i+1}</sub>" for _i in range(_data_transformed.shape[1])],
        color=_target,
        template="plotly_dark",
        opacity=0.5,
    )

    _fig_data_transformed.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig_data_transformed.data]
    for _i, _trace in enumerate(_fig_data_transformed.data):
        _trace.marker.line.color = "black"
        _trace.marker.line.width = 1
        _trace.marker.size = 4.5

    _data_transformed_component = mo.vstack(
        [
            mo.md("### 主成分スコア"),
            mo.hstack(
                [_df_data_transformed, _fig_data_transformed],
                gap=1,
                widths=[1, 2]
            )
        ],
        gap=1,
    )

    # 寄与率の可視化
    _fig_evr = px.bar(
        _df_evr,
        x="Principal Component",
        y="Explained Variance Ratio",
        title="Explained Variance Ratio by Principal Component",
    )

    _fig_evr.update_layout(
        bargap=0.5,         # 棒と棒の間の間隔（0〜1）
        bargroupgap=0.0      # グループ間の間隔（0で密着）
    )

    _evr_component = mo.vstack(
        [
            mo.md("### 寄与率の可視化"),
            mo.hstack([_df_evr, _fig_evr], gap=1, widths=[1, 2]),
        ],
        gap=1
    )

    # 因子負荷量の可視化
    _fig_factor_loading = px.bar(
        _df_factor_loading,
        x="Principal Component",
        y="Factor Loading",
        color="Feature",
        barmode="group",     # 横に並べる
        height=400,
        title="Factor Loadings by Feature and Principal Component"
    )


    _fig_factor_loading.update_layout(
        bargap=0.5,         # 棒と棒の間の間隔（0〜1）
        bargroupgap=0.0      # グループ間の間隔（0で密着）
    )

    _factor_loading_component = mo.vstack(
        [
            mo.md("### 因子負荷量の可視化"),
            mo.hstack([_df_factor_loading, _fig_factor_loading], gap=1, widths=[1, 2]),
        ],
        gap=1
    )

    _subtitle = mo.md(
        """
        ## 6. アヤメデータセットに関する主成分分析

        最後に、アヤメのデータセットに含まれる、すべての特徴を使用して主成分分析を実施する。  
        2次元での分析と同様の可視化を行い、結果を下記に示す。
        """
    )

    pca_iris_component = mo.vstack(
        [
            _subtitle,
            _dataset_component,
            _data_transformed_component,
            _evr_component,
            _factor_loading_component,
        ],
        gap=3
    )

    # pca_iris_component
    return (pca_iris_component,)


@app.cell
def _(
    evr_component,
    factor_loading_component,
    iris_dataset_component,
    maximize_var_component,
    mo,
    page_title,
    pca_iris_component,
    standardize_component,
):
    # コンポーネントの連結
    mo.vstack(
        [
            page_title,
            iris_dataset_component,
            standardize_component,
            maximize_var_component,
            evr_component,
            factor_loading_component,
            pca_iris_component,
        ],
        gap=5
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
