#import "../lib.typ": *

#show: simple_report.with(
  title: [Parallel Scientific Computing: GPU & MPI],
  author: "Toby Davis (cltz62)",
  institution: "Durham University",
  bibliography: bibliography("refs.bib"),
)

= Vector Triad

// Graph scaling N
// Report effective memory bandwidth
// Explain

#{
  let xs = (0, 1, 2, 3, 4)

  v(5mm)

  [
    #figure(
      placement: none,
      scope: "column",
      lq.diagram(
        title: [Vector Triad Memory Bandwidth],
        xlabel: $x$,
        ylabel: $y$,
        width: 100%,

        lq.plot(xs, (3, 5, 4, 2, 3), mark: "s", label: [A]),
        lq.plot(
          xs,
          x => 2 * calc.cos(x) + 3,
          mark: "o",
          label: [B],
        ),
      ),
      caption: [A caption for this figure],
    )
  ]
}

= Reduction

// Explain reduction algorithm
// Compare GPU results to CPU
// Pick and justify floating point tolerance

// Graph?

= General Matrix Multiply (GEMM)

== Problem Size Scaling

// Grpah with performance against problem size for each algorithm
// Table with results?
// Explain

== Strong Scaling

// Fix problem size
// Vary number of ranks
// Plot runtime vs ranks
// Discuss scaling
// Explain impact of MPI communication relative to GPU computation

== Profiled Results

// Profile at least one multi-rank run
// Identify:
//  - GPU Idle time
//  - What limits scaling?
//  - What would you optimise next?

#pagebreak()

= Introduction

Scientific writing is a crucial part of the research process, allowing
researchers to share their findings with the wider scientific community.
However, the process of typesetting scientific documents can often be
a frustrating and time-consuming affair, particularly when using outdated tools
such as LaTeX. Despite being over 30 years old, it remains a popular choice for
scientific writing due to its power and flexibility. However, it also comes with
a steep learning curve, complex syntax, and long compile times, leading to
frustration and despair for many researchers @netwok2020 @netwok2022.

#{
  let xs = (0, 1, 2, 3, 4)

  v(5mm)

  [
    #figure(
      placement: top,
      scope: "parent",
      lq.diagram(
        title: [Precious data],
        xlabel: $x$,
        ylabel: $y$,
        width: 100%,

        lq.plot(xs, (3, 5, 4, 2, 3), mark: "s", label: [A]),
        lq.plot(
          xs,
          x => 2 * calc.cos(x) + 3,
          mark: "o",
          label: [B],
        ),
      ),
      caption: [A caption for this figure],
    )<lovely_graph>
  ]

  align(center, lq.diagram(
    title: [Precious data],
    xlabel: $x$,
    ylabel: $y$,

    lq.plot(xs, (3, 5, 4, 2, 3), mark: "s", label: [A]),
    lq.plot(
      xs,
      x => 2 * calc.cos(x) + 3,
      mark: "o",
      label: [B],
    ),
  ))

  v(2mm)

  align(center, lq.diagram(
    width: 4cm,
    height: 4cm,
    lq.contour(
      lq.linspace(-5.0, 5.0, num: 20),
      lq.linspace(-5, 5, num: 12),
      (x, y) => x * y,
      map: color.map.viridis,
      levels: 20,
    ),
  ))

  v(5mm)
}

+ This
  + Sub item
  + Sub item
    + Sub sub item
+ Is
+ A
+ List

== Subsection

```rust
let levels = counter(heading).get()
let deepest = if levels != () {
  levels.last()
} else {
  1
}
```

Hello

This is some text in a subsection.

=== Subsubsection

This is a subsubsection.

== Paper overview
In this paper we introduce Typst, a new typesetting system designed to
streamline the scientific writing process and provide researchers with a fast,
efficient, and easy-to-use alternative to existing systems. Our goal is to shake
up the status quo and offer researchers a better way to approach scientific
writing.

#definition(
  title: "Pythagoras' Theorem",
  block(
    [For every right triangle, the sum of the squares of the legs of the triangle
      is equal to the square of the hypotenuse.

      For legs $a$ and $b$ and hypotenuse $c$:

      #math.equation(numbering: none, block: true)[
        $
          a^2 + b^2 = c^2
        $
      ]
    ],
  ),
  // footer: "Footer text",
)

#theorem(
  block(
    [
      For every right triangle, the sum of the squares of the legs of the
      triangle is equal to the square of the hypotenuse.

      For legs $a$ and $b$ and hypotenuse $c$:

      #math.equation(numbering: none, block: true)[
        $
          a^2 + b^2 = c^2
        $
      ]
    ],
  ),
  footer: "Footer text",
)

#proof(
  block(
    [
      This is a proof.
    ],
  ),
  footer: "Footer text",
)

#note(
  title: [A note],
  [
    Hello, world!
  ],
)

#cetz.canvas(length: 2cm, {
  import cetz.draw: *
  let phi = (1 + calc.sqrt(5)) / 2

  ortho({
    hide({
      line(
        (-phi, -1, 0),
        (-phi, 1, 0),
        (phi, 1, 0),
        (phi, -1, 0),
        close: true,
        name: "xy",
      )
      line(
        (-1, 0, -phi),
        (1, 0, -phi),
        (1, 0, phi),
        (-1, 0, phi),
        close: true,
        name: "xz",
      )
      line(
        (0, -phi, -1),
        (0, -phi, 1),
        (0, phi, 1),
        (0, phi, -1),
        close: true,
        name: "yz",
      )
    })

    intersections("a", "yz", "xy")
    intersections("b", "xz", "yz")
    intersections("c", "xy", "xz")

    set-style(stroke: (thickness: 0.5pt, cap: "round", join: "round"))
    line((0, 0, 0), "c.1", (phi, 1, 0), (phi, -1, 0), "c.3")
    line("c.0", (-phi, 1, 0), "a.2")
    line((0, 0, 0), "b.1", (1, 0, phi), (-1, 0, phi), "b.3")
    line("b.0", (1, 0, -phi), "c.2")
    line((0, 0, 0), "a.1", (0, phi, 1), (0, phi, -1), "a.3")
    line("a.0", (0, -phi, 1), "b.2")

    anchor("A", (0, phi, 1))
    content("A", [$A$], anchor: "north", padding: .1)
    anchor("B", (-1, 0, phi))
    content("B", [$B$], anchor: "south", padding: .1)
    anchor("C", (1, 0, phi))
    content("C", [$C$], anchor: "south", padding: .1)
    line("A", "B", stroke: (dash: "dashed"))
    line("A", "C", stroke: (dash: "dashed"))
  })
})

By leveraging advanced algorithms and a user-friendly interface, Typst offers
several advantages over existing typesetting systems, including faster document
creation, simplified syntax, and increased ease-of-use.

#box(
  height: 8em,
  align(
    center + horizon,
    $
      markul(p_i, tag: #<p>) = markrect(
        exp(- mark(beta, tag: #<beta>, color: #red) mark(E_i, tag: #<E>, color: #green)),
        tag: #<Boltzmann>, color: #blue,
      ) / markhl(sum_j exp(- beta E_j), tag: #<Z>)
      #annot(<p>, pos: bottom + left)[Probability of \ state $i$]
      #annot(<beta>, pos: top + left, dy: -1.5em, leader-connect: "elbow")[Inverse temperature]
      #annot(<E>, pos: top + right, dy: -1em)[Energy]
      #annot(<Boltzmann>, pos: top + left)[Boltzmann factor]
      #annot(<Z>)[Partition function]
    $,
  ),
)

To demonstrate the potential of Typst, we conducted a series of experiments
comparing it to other popular typesetting systems, including LaTeX. Our findings
suggest that Typst offers several benefits for scientific writing, particularly
for novice users who may struggle with the complexities of LaTeX. Additionally,
we demonstrate that Typst offers advanced features for experienced users,
allowing for greater customization and flexibility in document creation.

This links to @lovely_graph
This links to #link(<lovely_graph>, [the same graph])

Overall, we believe that Typst represents a significant step forward in the
field of scientific writing and typesetting, providing researchers with a
valuable tool to streamline their workflow and focus on what really matters:
their research. In the following sections, we will introduce Typst in more
detail and provide evidence for its superiority over other typesetting systems
in a variety of scenarios.

#no_indent[
  This paragraph is not indented
]

#codly(highlights: (
  (line: 4, start: 3, end: none, fill: red),
  (line: 5, start: 12, end: 21, fill: green, tag: "(a)"),
  (line: 5, start: 25, fill: blue, tag: "(b)"),
))
```py
def fib(n):
  if n <= 1:
    return n
  else:
    return fib(n - 1) + fib(n - 2)
print(fib(25))
```

= Methods <sec:methods>
#lorem(45)

$ a + b = gamma $ <eq:gamma>

#lorem(80)

#figure(
  scope: "column",
  placement: none,
  circle(radius: 15pt),
  caption: [A circle representing the Sun.],
) <fig:sun>

In @fig:sun you can see a common representation of the Sun, which is a star that
is located at the center of the solar system.

#lorem(120)

#figure(
  caption: [The Planets of the Solar System and Their Average Distance from the Sun],
  placement: top,
  table(
    // Table styling is not mandated by the IEEE. Feel free to adjust these
    // settings and potentially move them into a set rule.
    columns: (6em, auto),
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[Planet][Distance (million km)],
    [Mercury], [57.9],
    [Venus], [108.2],
    [Earth], [149.6],
    [Mars], [227.9],
    [Jupiter], [778.6],
    [Saturn], [1,433.5],
    [Uranus], [2,872.5],
    [Neptune], [4,495.1],
  ),
) <tab:planets>

In @tab:planets, you see the planets of the solar system and their average
distance from the Sun. The distances were calculated with @eq:gamma that we
presented in @sec:methods.

#lorem(240)

#lorem(240)

= This is a section

Some stuff.

== This is a subsection

Some more stuff

=== This is a sub-subsection

Even more stuff

==== A ludicrously deep subsection heading

What does this even mean?

Is this indented? What if I have a lot of text here? This is soem random stuff
to fill up space and get a paragraph break.

Is this indented? What if I have a lot of text here? This is soem random stuff
to fill up space and get a paragraph break.

==== Another section

What does this even mean?

Is this indented? What if I have a lot of text here? This is soem random stuff
to fill up space and get a paragraph break.

Is this indented? What if I have a lot of text here? This is soem random stuff
to fill up space and get a paragraph break.

===== A ludicrously deep subsection heading

Woah... We're going even deeper

Is this indented??

===== Another deep subsection heading

Oh all good. It seems like we've stopped here.

$y = a x + b$

$
  a^2 + b^2 = c^2
$

#eq_no_num[$
  a^2 + b^2 = c^2
$]

#import "@preview/game-theoryst:0.1.0": *

#figure(
  scope: "column",
  placement: none,

  table(
    align: center,
    columns: 2,
    stroke: none,
    nfg(
      players: ($P_1$, $P_2$),
      s1: ("X", "Y"),
      s2: ("X", "Y"),
      [$(3, 3)$],
      [$(1, 5)$],
      [$(5, 1)$],
      [$(0, 0)$],
    ),
    nfg(
      players: ($P_1$, $P_2$),
      s1: ($alpha$, $beta$),
      s2: ($alpha$, $beta$),
      [$(2, 1)$],
      [$(0, 5)$],
      [$(3, -2)$],
      [$(1, -1)$],
    ),
  ),
  caption: [
    #par([
      Left: Not the same as the Prisoner's Dilemma, as there is not always a
      benefit to switching to Y.
    ])

    #par([
      Right: Though it has different values, the players always deviate to
      playing $beta$.
    ])
  ],
)

