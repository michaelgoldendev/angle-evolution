import Escher: @api, render

desc = md"""
# Hello, World!
You can write any **Markdown** text and place it in the middle of other
tiles.
""" |> Escher.pad(1em)

main(window) = begin
    push!(window.assets, "layout2")
    push!(window.assets, "icons")
    push!(window.assets, "widgets")

    t, p = wire(tabs([hbox(icon("home"), hskip(1em), "Homea"),
                      hbox(icon("info-outline"), hskip(1em),  "Notifications"),
                      hbox(icon("settings"), hskip(1em), "Settings")]),
                pages([desc, "Notification tab content", "Settings tab content"]), :tabschannel, :selected)

    vbox(toolbar([iconbutton("face"), "My App", flex(), iconbutton("search")]),
         maxwidth(30em, t),
         Escher.pad(1em, p))
end
#=

function main(window)
    #txt = tex("T = 2\\pi\\sqrt{L\\over g}")
    #fillcolor("#eeb", fontcolor("#499", pad(5mm, txt)))
    #Elem(:div, innerHTML="$(readall("benchmarks_scatteplot_phi.svg"))")
    iterations = Input(0) # The angle at any given time
    connected_slider = subscribe(slider(0:7), iterations)
end=#
