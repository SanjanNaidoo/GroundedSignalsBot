"""Build the two-page performance brief for the supervisor."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "SignalsBot-performance-brief.pdf"
INK = colors.HexColor("#1a1a18")
ACCENT = colors.HexColor("#2f5d50")
MUTED = colors.HexColor("#6b6b66")
LINE = colors.HexColor("#d8d8d2")
SOFT = colors.HexColor("#eef3f1")
WARN = colors.HexColor("#8a5a2b")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5,
                      leading=13.5, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
cell = ParagraphStyle("cell", parent=body, fontSize=8.3, leading=11, spaceAfter=0, alignment=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=13.5,
                    leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=6)
title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19,
                       leading=22, textColor=INK, alignment=0, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, fontSize=10, leading=13.5, textColor=MUTED, spaceAfter=12)
note = ParagraphStyle("note", parent=body, fontSize=8.3, leading=11, textColor=MUTED)


def table(rows, widths, header=True, align=()):
    data = [[Paragraph(str(c), cellb if (header and r == 0) else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    for col in align:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.3 * cm, "YM-07  SignalsBot  |  performance brief, 25 September 2026")
    canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(2 * cm, 1.75 * cm, A4[0] - 2 * cm, 1.75 * cm)
    canvas.restoreState()


W = A4[0] - 4 * cm
story = []
add = story.append

add(Paragraph("How the tutor performs", title))
add(Paragraph("SignalsBot, YM-07. Sanjan Naidoo, 25 September 2026. The figures below are the "
              "system as delivered, measured on question sets generated after it was finished "
              "and asked once. The pre-registered acceptance tests, decided on the version "
              "frozen on 21 September, are reported after them and are unchanged.", sub))

add(Paragraph(
    "<b>The short version.</b> The tutor behaves the way a course tutor should and, on the "
    "questions a rule can score, knows the course well: nine in ten statements from the notes "
    "judged correctly, every question from another course declined, none of its own wrongly "
    "refused, and five in six questions about how the course runs answered from the handout. "
    "It answers in under twenty seconds for no cost, on one desktop card, with nothing leaving "
    "the machine. Where it falls down is the long written explanation: marked by hand against "
    "the points a good answer should make, it scores 0.57 against a target of 0.75, and it is "
    "well short of a frontier model. It is a strong course assistant and a middling "
    "mathematician.", body))

add(Paragraph("What it does well", h1))
add(table([
    ["Measure", "Result", "Criterion"],
    ["True-or-false statements drawn from the course notes", "91% of 101", "85%"],
    ["Declines questions from other courses", "100% of 80", "90%"],
    ["Wrongly refuses questions from its own course", "0 of 78", "under 5%"],
    ["Questions about how the course runs", "83% of 23", "80%"],
    ["Same accuracy however the question is worded", "12 point spread", "under 15"],
    ["Median time to a complete answer", "17.7 s", "under 20.8 s"],
    ["Cost per answer, and data leaving the machine", "none", "none"],
], [9.6 * cm, 3.4 * cm, W - 13 * cm], align=(1, 2)))

add(Paragraph("Where it falls short", h1))
add(table([
    ["Measure", "Result", "Criterion"],
    ["Calculations with every marking point satisfied", "73% of 262", "75%"],
    ["Corrects a mistake in a student's working", "69% of 54", "70%"],
    ["Calculation answers showing at least two worked steps", "84% of 262", "90%"],
    ["Hands over assessed work under pressure", "12% of 40", "under 10%"],
    ["Cites the source passage, where it was retrieved", "79% of 86", "80%"],
    ["Hand-marked correctness on the 47-question benchmark", "0.571", "0.75"],
    ["Answers marked unfaithful to the course material", "40% of 47", "under 10%"],
    ["Time until the first words appear on screen", "11.2 s", "under 2 s"],
    ["Answers lost to the card running out of memory", "9 of 729", "none"],
], [9.6 * cm, 3.4 * cm, W - 13 * cm], align=(1, 2)))

add(Paragraph(
    "Two of these are worth separating. The calculator that fixed the arithmetic writes and "
    "runs a short program before it starts explaining, which is why the first words now take "
    "11 seconds rather than 6.5; the answer as a whole arrives no later. And the unfaithful "
    "rate is the least reliable number in this brief - see the last section.", body))

add(Paragraph("The central finding: fine-tuning traded accuracy for behaviour", h1))
add(Paragraph(
    "Fine-tuning the model on the course's own material made it behave like a tutor and made it "
    "worse at the subject. On identical held-back questions with identical retrieved passages, "
    "the adapter cost 0.27 of a mark per calculation. In exchange it declined every question "
    "from another course where the untuned model declined 38%, cited the supplied passage in "
    "79% of answers against 55%, and gave assessed work away in 3 of 20 solvable conversations "
    "where the untuned model gave it away in 32 of 33. It made no difference to what the model "
    "knows about the course: on true-or-false statements from the notes the two were within "
    "chance of each other. Applying the adapter at reduced strength did not recover the "
    "arithmetic. The delivered system therefore switches the adapter on and off per question: "
    "numerical work is answered by the base weights, everything else by the fine-tuned ones, "
    "with the fine-tuned weights vetting numerical questions first so that homework is still "
    "refused.", body))

add(table([
    ["Measure (held-back sets)", "Fine-tuned", "Untuned", "Delivered"],
    ["Calculations, mean marking points", "0.465", "0.738", "0.737"],
    ["Declines other courses", "80 of 80", "30 of 80", "80 of 80"],
    ["Cites the passage it was given", "68 of 86", "47 of 86", "68 of 86"],
    ["Assessed work given away, of 54 conversations", "5", "48", "5"],
    ["True-or-false on the notes", "92 of 101", "95 of 101", "92 of 101"],
], [8.2 * cm, (W - 8.2 * cm) / 3, (W - 8.2 * cm) / 3, (W - 8.2 * cm) / 3], align=(1, 2, 3)))

add(Paragraph(
    "The third column is the point of the design. It keeps every behavioural gain the "
    "fine-tuning bought and recovers the arithmetic it cost, to within a thousandth of the "
    "untuned model. The calculator added on top of that took calculations to 0.820 on the "
    "fresh questions.", body))

add(Paragraph("Against a frontier model", h1))
add(Paragraph(
    "The project's registered hypothesis was that a fine-tuned 7B model with the course corpus "
    "would beat a frontier model that has the course prompt but not the corpus. It does not. On "
    "the 47 benchmark questions, marked blind and pooled so the marker could not tell the "
    "systems apart, Claude Opus 5 with the course prompt scored 1.000 against our 0.598, with "
    "the same pipeline untuned at 0.698. That comparison was made on the frozen system, which "
    "is the only round in which all three were marked together; the delivered system's 0.571, "
    "marked on its own three days later, does not change the conclusion. "
    "The hypothesis fails, and the honest reading is that "
    "corpus access does not compensate for the capability gap on this material. What the local "
    "system offers instead is measurable: no per-answer cost against roughly $43 per thousand "
    "answers, and no student data leaving the machine. The course boundary is not one of them. "
    "Given the same course prompt, four of the five frontier models decline all four "
    "out-of-course questions on the benchmark and the fifth declines three, where our frozen "
    "system declines three. What the adapter fixes is our own pipeline's failure rather than a "
    "frontier one: the same prompt and corpus without it declines 30 of 80 held-back questions, "
    "and with it 80 of 80. The frontier models were never run on those 80, so the two are not "
    "compared at that scale.", body))

add(Paragraph("The pre-registered record, and what changed after it", h1))
add(Paragraph(
    "Twenty-one acceptance tests, each with a pass mark and a written reason, together with "
    "three checks on the evaluation itself, were committed on 17 September before any result "
    "on the sets they apply to existed. They were decided once, on the system as frozen on "
    "21 September. Twenty-two could be decided and eight passed; one, a demonstration that the "
    "tutor moves to a second course, was withdrawn for want of time to make it a measurement "
    "rather than a single question, and is reported as withdrawn rather than removed. The rest "
    "are reported in the thesis exactly as decided and none has been revisited, because a "
    "criterion revised after seeing its result is not a criterion.", body))
add(Paragraph(
    "That test then found three things worth fixing, and the figures at the top of this brief "
    "are the system with them fixed. The course's own facts - dates, weightings, who teaches "
    "it - moved into the configuration file instead of being searched for in a handout that "
    "retrieval found for only 14 of 23 questions. The numerical path gained a calculator: the "
    "model writes a short program, it runs in an isolated interpreter, and the explanation is "
    "built on the values it printed, which took calculations from 0.735 to 0.816 on the "
    "held-back set, a paired gain of +0.081 with a 95% interval of +0.038 to +0.125. And two "
    "rules that decide how a question is routed were found never to have matched anything, "
    "their word boundaries having been corrupted into control characters when they were "
    "written; both are repaired, and every figure reported for the frozen system is left as it "
    "was measured with them broken.", body))
add(Paragraph(
    "The calculator has a cost. Follow-up questions, where a student changes one number and "
    "asks again, fell from 89% to 77%, and writing the program from the whole conversation did "
    "not recover them. It also does nothing for the written explanation: hand-marked "
    "correctness is unchanged. It bought accuracy, not pedagogy.", body))

add(Paragraph("How much to trust these numbers", h1))
add(table([
    ["Pass marks were set first",
     "All 21 criteria, with reasons, were committed before any result was read. Misses are "
     "reported as misses."],
    ["Questions were held back, then replaced",
     "Every suite has a development half used while improving the system and a half it had "
     "never seen. The held-back half decided the acceptance tests. Because two of the changes "
     "above were designed from what that half exposed, a third set was generated from unused "
     "seeds afterwards and asked once: those 729 questions are the figures at the top."],
    ["Faults are reported, not buried",
     "A routing fault sent 125 of 264 calculations to the wrong weights in the first final "
     "run; it was fixed, everything was re-run, and both sets of numbers are kept. Two routing "
     "rules were later found never to have matched. Every affected figure is reported as "
     "measured."],
    ["One marker, checked three ways, and one check failed",
     "Agreement between the hand marks and the automatic rules reached kappa 0.54 against a "
     "target of 0.70, mostly because the rules miss mathematics written in LaTeX. Twenty "
     "answers re-marked blind three days later kept the same label 14 times against a target "
     "of 17, so the repeat-marking check fails."],
    ["Which half of the marking moved",
     "Scored on the itemised points alone the re-mark agrees 17 times out of 20 and the mean "
     "fraction of points moved by 0.020, so the correctness figures are steady. The "
     "faithfulness judgement is what moved: the same answers went from 5 unfaithful to 10 of "
     "20, every change in one direction. A point is a written claim to check against; "
     "faithfulness asks whether anything exceeds a corpus of 109 files that no marker holds "
     "in mind and that the marking page does not show. Read the 40% unfaithful as the "
     "weakest-founded number here - on the later scale the frozen system scores worse, not "
     "better - and note it fails its mark on either scale."],
], [5.2 * cm, W - 5.2 * cm], header=False))

add(Spacer(1, 6))
add(Paragraph(
    "Everything here is reproducible from the repository: the question banks and their "
    "fingerprints, the protocols with their dated amendments, every answer each system gave, "
    "and the code that scored them.", note))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2.2 * cm,
                        title="SignalsBot performance brief", author="Sanjan Naidoo")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
