import matplotlib
  # Use a non-interactive backend

import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt

output_file = "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-p(R4,CPE4).png"

with schemdraw.Drawing() as d:
    # Add R0
    d += elm.Resistor().label("R0")
    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R1").right()
    d += elm.Line().down(d.unit * 0.5)

    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.CPE().label("CPE1").right()
    d += elm.Line().up(d.unit * 0.5)

    d += elm.Line().right(d.unit * 0.5)

    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R2").right()
    d += elm.Line().down(d.unit * 0.5)

    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.CPE().label("CPE2").right()
    d += elm.Line().up(d.unit * 0.5)

    d += elm.Line().right(d.unit * 0.5)

    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R3").right()
    d += elm.Line().down(d.unit * 0.5)

    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.CPE().label("CPE3").right()
    d += elm.Line().up(d.unit * 0.5)

    d += elm.Line().right(d.unit * 0.5)

    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R4").right()
    d += elm.Line().down(d.unit * 0.5)

    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.CPE().label("CPE4").right()
    d += elm.Line().up(d.unit * 0.5)

    d += elm.Line().right(d.unit * 0.5)
    
    d.draw()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  # Explicit Matplotlib save
