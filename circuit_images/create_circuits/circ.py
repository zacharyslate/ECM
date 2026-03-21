import schemdraw
import schemdraw.elements as elm

output_file = "R0-p(R1,C1)-p(R2,C2)"

with schemdraw.Drawing() as d:
    # Add R0
    d += elm.Resistor().label("R0").right()

    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R1").right()
    d += elm.Line().down(d.unit * 0.5)
    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.Capacitor().label("C1").right()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Line().right(d.unit * 0.5)

#    d.push()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Resistor().label("R2").right()
#    d += elm.Line().down(d.unit * 0.5)
#    d.pop()
#    d += elm.Line().down(d.unit * 0.5)
#    d += elm.CPE().label("CPE2").right()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Line().right(d.unit * 0.5)

#    d.push()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Resistor().label("R3").right()
#    d += elm.Line().down(d.unit * 0.5)
#    d.pop()
#    d += elm.Line().down(d.unit * 0.5)
#    d += elm.CPE().label("CPE3").right()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Line().right(d.unit * 0.5)

#    d.push()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Resistor().label("R4").right()
#    d += elm.Line().down(d.unit * 0.5)
#    d.pop()
#    d += elm.Line().down(d.unit * 0.5)
#    d += elm.CPE().label("CPE4").right()
#    d += elm.Line().up(d.unit * 0.5)
#    d += elm.Line().right(d.unit * 0.5)

    d.push()
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Resistor().label("R2").right()
    d += elm.Line().down(d.unit * 0.5)
    d.pop()
    d += elm.Line().down(d.unit * 0.5)
    d += elm.Capacitor().label("C2").right()
    #d += elm.ResistorIEC().label(r"${War}$", fontsize=14, ofst=-0.5).right().label("W2")
    d += elm.Line().up(d.unit * 0.5)
    d += elm.Line().right(d.unit * 0.5)

#    d += elm.Resistor().label("R6").right()

    d.push()
    d.draw()
    d.save(output_file)
