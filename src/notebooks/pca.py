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
    return mo, np, pd, px


@app.cell
def _(mo):
    # ページタイトル
    page_title = mo.md("# 主成分分析 (PCA)")
    page_title
    return


@app.cell
def _(mo, np, pd):
    # データセットの読み込み
    df_iris = pd.read_csv("datasets/iris.csv")

    data = df_iris.drop(columns="Species")
    target = df_iris["Species"]
    dimensions = data.columns
    unique_target = np.unique(target)

    # スイッチ作成
    switches_dimention = mo.ui.array(
        [mo.ui.switch(value=True, label=_dimension) for _dimension in dimensions],
        label="demensions"
    )
    switches_target = mo.ui.array(
        [mo.ui.switch(value=True, label=_target) for _target in unique_target],
        label="target"
    )
    return (
        data,
        df_iris,
        dimensions,
        switches_dimention,
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
    switches_dimention,
    switches_target,
    target,
    unique_target,
):
    # スイッチによるディメンション, ターゲットの選択
    _dimentions_selected = dimensions[switches_dimention.value]
    _target_selected = unique_target[switches_target.value]

    _data = data[_dimentions_selected][target.isin(_target_selected)]
    _target = target[target.isin(_target_selected)]

    # 散布図の作成
    _fig = px.scatter_matrix(
        _data,
        dimensions=_dimentions_selected,
        color=_target,
        template="plotly_dark",
        # width=700,
        # height=600
    )

    _fig.update_layout(
        legend_title_text="Species"
    )

    _colors = [trace.marker.color for trace in _fig.data]
    for _i, _trace in enumerate(_fig.data):
        _trace.marker.color = "rgba(0,0,0,0)"
        _trace.marker.line.color = _colors[_i]
        _trace.marker.line.width = 1
        _trace.marker.size = 4.5

    # コンポーネント作成
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
            mo.md("""
    ### 詳細

    植物のアヤメの各品種別の特徴をまとめたデータ

    特徴の詳細は下記の通り

    - SepalLength  : がく片の長さ
    - SepalWidth   : がく片の幅
    - PetalLength  : 花弁の長さ
    - PetalLength  : 花弁の幅
    - Species      : アヤメの品種
            """)
        ],
        gap=3,
        widths=[2, 1],
    )

    _switches = mo.vstack(
        [
            switches_dimention,
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
            mo.md("## 使用データセット"),
            _table_description,
            _scatter_matrix_with_switch,
        ],
        gap=2,
    )

    iris_dataset_component
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
