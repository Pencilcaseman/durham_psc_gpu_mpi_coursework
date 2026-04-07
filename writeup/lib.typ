// Packages
#import "@preview/cetz:0.4.2" as cetz
#import "@preview/codly-languages:0.1.8": *
#import "@preview/codly:1.3.0": *
#import "@preview/lilaq:0.5.0" as lq
#import "@preview/mannot:0.3.0": *
#import "@preview/physica:0.9.6": *
#import "@preview/showybox:2.0.4" as showybox

#let simple_report(
  title: none,
  author: none,
  institution: none,
  paper-size: "a4",
  bibliography: none,
  body,
) = {
  // Main document configuration
  set document(title: title, author: author)

  // Font config
  set text(size: 10pt, spacing: .35em)

  set figure(scope: "parent", placement: top)

  // Code blocks
  show raw: set text(
    font: "Monaspace Neon",
    ligatures: true,
    size: 1em,
    spacing: 100%,
  )

  set columns(gutter: 12pt)
  set page(
    numbering: none,
    footer: none,
    columns: 2,
    paper: paper-size,
    // The margins depend on the paper size.
    margin: if paper-size == "a4" {
      (x: 1.25cm, top: 1.5cm, bottom: 1.5cm)
    } else {
      (
        x: (50pt / 216mm) * 100%,
        top: (55pt / 279mm) * 100%,
        bottom: (64pt / 279mm) * 100%,
      )
    },
  )

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)")
  show math.equation: set block(spacing: 0.65em)
  set math.mat(delim: "[")

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  set heading(numbering: "I.A.a)")
  show heading: it => {
    set text(10pt, weight: 400)

    let level = counter(heading).get()
    let last_level = if level != () {
      level.last()
    } else {
      1
    }

    if it.level == 1 {
      let exclude = (
        it.body
          in (
            [Acknowledgment],
            [Acknowledgement],
            [Acknowledgments],
            [Acknowledgements],
          )
      )

      set align(center)
      set text(if exclude { 10pt } else { 14pt })
      show: block.with(above: 20pt, below: 15pt, sticky: true)
      show: smallcaps

      if it.numbering != none and not exclude {
        numbering("I.", last_level)
        h(7pt, weak: true)
      }

      it.body
    } else if it.level == 2 {
      // Second-level headings are run-ins.
      set align(center)
      set text(size: 12pt)
      show: block.with(spacing: 15pt, sticky: true)
      show: smallcaps

      if it.numbering != none {
        numbering("1.", last_level)
        h(7pt, weak: true)
      }

      it.body
    } else if it.level == 3 {
      set align(center)
      set text(size: 11pt)
      show: smallcaps
      show: block.with(spacing: 10pt, sticky: true)
      numbering("A.", last_level)
      h(7pt, weak: true)

      it.body
    } else {
      set align(left)
      set text(size: 10pt)
      show: smallcaps

      block[
        #numbering("I.I.I.I.I.I.I)", ..level.slice(3))
        #h(7pt, weak: true)
        #it.body
      ]
    }
  }

  // Style links
  show link: it => {
    set text(fill: blue.darken(30%), style: "italic")
    it
  }

  // Style bibliography.
  show std.bibliography: set text(8pt)
  show std.bibliography: set block(spacing: 0.5em)
  set std.bibliography(title: text(10pt)[References], style: "ieee")

  show: codly-init.with()
  codly(languages: codly-languages)

  // Display authors on the title page. We want to span the whole page, hence
  // the float at the parent scope
  place(
    top,
    float: true,
    scope: "parent",
    clearance: 10pt,
    {
      line(
        angle: 0rad,
        length: 100%,
        stroke: stroke(thickness: 0.4mm),
      )

      set par(leading: 5pt, spacing: 10pt)

      {
        set text(size: 24pt)
        title
      }

      v(2mm)

      {
        set text(size: 18pt)
        author
      }

      parbreak()

      grid(
        rows: 1,
        columns: (1fr, 1fr),
        align: (left, right),
        {
          set text(size: 13pt, style: "italic")
          institution
        },
        {
          set text(size: 13pt)
          text(datetime.today().display("[day] [month repr:long] [year]"))
        },
      )

      line(
        angle: 0rad,
        length: 100%,
        stroke: stroke(thickness: 0.4mm),
      )

      v(0.5cm)
    },
  )

  set par(
    justify: true,
    first-line-indent: (amount: 1em, all: false),
    spacing: 0.75em,
    leading: 0.5em,
  )

  // Display the paper's contents.
  body

  // Display bibliography.
  bibliography
}

#let eq_no_num(body) = {
  math.equation(block: true, numbering: none, body)
}

#let no_indent(body) = context {
  let prev_indent = par.first-line-indent
  set par(first-line-indent: 0em)

  body

  set par(first-line-indent: prev_indent)
}

#let colored_titlebox(title: [Definition], footer: "", color: color, body) = {
  h(5mm)

  if title == none {
    showybox.showybox(
      frame: (
        body-color: color.lighten(90%),
        footer-color: color.lighten(75%),
        border-color: color.darken(50%),
        radius: (rest: 1mm),
      ),
      footer: if footer == "" { "" } else {
        text(
          emph(footer),
        )
      },
      body,
    )
  } else {
    showybox.showybox(
      title-style: (
        boxed-style: (
          anchor: (
            x: left,
            y: horizon,
          ),
          radius: (rest: 1mm),
        ),
      ),
      frame: (
        title-color: color.darken(50%),
        body-color: color.lighten(90%),
        footer-color: color.lighten(75%),
        border-color: color.darken(50%),
        radius: (rest: 1mm),
      ),
      title: title,
      footer: if footer == "" { "" } else {
        text(
          emph(footer),
        )
      },
      body,
    )
  }
}

#let definition(title: [Definition], footer: "", body) = {
  colored_titlebox(title: title, footer: footer, color: color.green, body)
}

#let theorem(title: [Theorem], footer: "", body) = {
  colored_titlebox(title: title, footer: footer, color: color.purple, body)
}

#let proof(title: [Proof], footer: "", body) = {
  colored_titlebox(title: title, footer: footer, color: color.blue, body)
}

#let note(title: [Note], footer: "", body) = {
  colored_titlebox(title: title, footer: footer, color: color.eastern, body)
}
