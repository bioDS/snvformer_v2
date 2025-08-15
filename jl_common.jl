using ColorSchemes
using Gadfly, Compose

function plot_roc(filename, fpr, tpr, auroc)
    colorscheme = ColorSchemes.seaborn_bright
    line = layer(x=0:1, y=0:1, Geom.line, color=[colorscheme.colors[1]])
    curve = layer(x=fpr, y=tpr, Geom.line, color=[colorscheme.colors[2]])
    key = Guide.manual_color_key("", [Printf.@sprintf("Classifier (auc = %.3f)", auroc)], [colorscheme.colors[2]])
    mykey = render(key, Theme(), Gadfly.Aesthetics())[1].ctxs[1]

    p = Gadfly.plot(curve, line, Guide.xlabel("False Positive Rate"), Guide.ylabel("True Positive Rate"),
        Guide.annotation(compose(Compose.context(),
            (Compose.context(0.15w, 0.38h), mykey))
        ),
        Theme(
            background_color = "white",
            default_color = "black",
            grid_color="grey90",
            # line_style=:solid,
            grid_line_style=:solid,
            key_position=:inside,
            major_label_font="DejaVuSans", #major_label_font_size=12pt,
            minor_label_font="DejaVuSans", #minor_label_font_size=10pt,
              key_title_font="DejaVuSans", #key_title_font_size=12pt,
              key_label_font="DejaVuSans", #key_label_font_size=10pt,
        ),
        Guide.xticks(ticks=[i/5.0 for i in 0:5]),
        Guide.yticks(ticks=[i/5.0 for i in 0:5]),
    )
    Gadfly.draw(Gadfly.PNG(Printf.@sprintf("%s.png", filename), 8cm, 8cm, dpi=300), p)
    Gadfly.draw(Gadfly.PDF(Printf.@sprintf("%s.pdf", filename), 8cm, 8cm), p)
end