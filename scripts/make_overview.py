"""Build the overview PDF: what the tutor does, how it is built, how it is tested."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = Path(r"C:\Users\User\Documents\SignalsBot\SignalsBot-overview.pdf")
INK = colors.HexColor("#1a1a18")
ACCENT = colors.HexColor("#2f5d50")
MUTED = colors.HexColor("#6b6b66")
LINE = colors.HexColor("#d8d8d2")
SOFT = colors.HexColor("#eef3f1")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5,
                      leading=13.5, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
small = ParagraphStyle("small", parent=body, fontSize=8.3, leading=11, spaceAfter=0)
cell = ParagraphStyle("cell", parent=small, alignment=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                    leading=18, textColor=ACCENT, spaceBefore=14, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                    leading=14, textColor=INK, spaceBefore=10, spaceAfter=5)
title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21,
                       leading=25, textColor=INK, alignment=0, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=14)
note = ParagraphStyle("note", parent=body, fontSize=8.5, leading=11.5, textColor=MUTED)


def table(rows, widths, header=True, align_right=()):
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(c), cellb if (header and r == 0) else cell) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 6),
             ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.3 * cm, "YM-07  SignalsBot  |  EEE4114F course tutor")
    canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(2 * cm, 1.75 * cm, A4[0] - 2 * cm, 1.75 * cm)
    canvas.restoreState()


W = A4[0] - 4 * cm
story = []
add = story.append

add(Paragraph("SignalsBot", title))
add(Paragraph("A course tutor for EEE4114F: what it does, how it is built, and how it is "
              "tested. Sanjan Naidoo, YM-07, 25 September 2026.", sub))

add(Paragraph(
    "SignalsBot answers questions about EEE4114F from the course's own material. It runs "
    "entirely on one desktop graphics card: no question a student types leaves the machine, "
    "and no answer costs anything to produce. It is built from a 7-billion-parameter "
    "open-weights model, a search index over the course notes, tutorials and past papers, and "
    "a fine-tuned adapter trained on that same material. Three things separate it from a "
    "general chatbot: it cites the passage behind each claim, it turns away questions from "
    "other courses, and it refuses to hand over work a student is going to submit for marks.", body))

add(table([
    ["Base model", "Qwen2.5-7B-Instruct, 4-bit, on one RTX 5070 Ti (16 GB)"],
    ["Course material", "335 passages from the notes, slides, tutorials, class tests and past papers"],
    ["Fine-tuning", "LoRA adapter, 982 training examples written from that material"],
    ["Cost per answer", "no API charge; about 8 to 25 seconds on the card"],
    ["Interface", "web chat: answers stream, mathematics is typeset, sources are listed"],
    ["Status", "frozen on 21 September 2026 and tested on held-back questions; two changes "
               "made after that test are measured separately"],
], [4.2 * cm, W - 4.2 * cm], header=False))

add(Paragraph("1. What it does", h1))
add(Paragraph(
    "The project brief asks for a tutor that explains core concepts, finds errors in "
    "calculations, shows mathematical working and answers questions about the course. Each of "
    "those is a feature below, and each has a test behind it. The figures are from questions "
    "the system had never seen, on the version frozen on 21 September; where a later change "
    "moved a figure, the later one follows it.", body))

add(table([
    ["What a student can do", "What the tutor does", "Measured"],
    ["Ask how something works",
     "Explains from the notes and cites the passage the claim comes from",
     "91% correct on 101 true-or-false statements drawn from the notes; cites the right "
     "passage in 79% of answers where retrieval supplied it"],
    ["Ask it to work something out",
     "Works the calculation through step by step and states the final value",
     "0.74 of the marking points on 264 generated calculations across eleven kinds of "
     "problem; 0.82 with the calculator added after the test"],
    ["Show it their own working",
     "Checks it, finds the mistake and gives the corrected value",
     "corrects 34 of 54 workings that contain a classic mistake; 38 with the calculator"],
    ["Change a number and ask again",
     "Carries the problem through the conversation without being told it again",
     "31 of 35 follow-ups right, where the first answer was right; 26 of 31 with the "
     "calculator"],
    ["Ask something from another course",
     "Says it is outside EEE4114F and declines, instead of answering from general knowledge",
     "declines 80 of 80 questions from other courses, while wrongly refusing 2 of 78 of its own"],
    ["Ask it to do work due for marks",
     "Names it as assessed work and scaffolds: points at the method, offers to check working",
     "gives the answer away in 5 of 33 conversations where it could have solved the problem"],
    ["Ask how the course runs",
     "Answers from the course's own record: dates, weightings, DP rules",
     "12 of 23 as tested; 22 of 23 once the course's facts moved into the configuration"],
], [3.6 * cm, 5.4 * cm, W - 9 * cm]))

add(Paragraph("2. How it is built", h1))
add(Paragraph(
    "An answer is produced in four steps. Everything course-specific is held in configuration "
    "rather than in code, so that moving the tutor to another course is intended to need a new "
    "configuration file and a new corpus rather than a new program. That is how it is built; "
    "the demonstration that would have tested it was withdrawn from scope, so it is not a "
    "claim this project verifies.", body))

add(table([
    ["1. Find the material",
     "The question is searched against 335 passages of course material with two methods at "
     "once, keyword matching and a sentence-embedding model, fused by rank. The best four "
     "passages go to the model. Four was chosen by measurement: over 150 questions written "
     "from known passages, this finds the source in the top four 75% of the time, against 53% "
     "for the simpler method it replaced."],
    ["2. Set the rules",
     "A course prompt is built from the configuration file. Each pedagogical choice in that "
     "file becomes a paragraph in the prompt: keep to the course, do not do assessed work, "
     "cite the passages, use the course's notation."],
    ["3. Choose the weights",
     "The fine-tuned adapter is switched on or off for each question. Numerical work is "
     "answered from the base weights; everything else from the fine-tuned ones. Before a "
     "numerical answer, the fine-tuned weights write a two-line opening, and if that opening "
     "declines the question or treats it as work to be handed in, their answer stands."],
    ["4. Answer",
     "The answer streams into the browser as it is written, with mathematics typeset, the "
     "passages it drew on listed beside it, and the time and cost of that answer shown."],
], [3.6 * cm, W - 3.6 * cm], header=False))

add(Paragraph(
    "Two things were added after the final test. The course's own facts - staff, dates, "
    "weightings, DP rules - are stated in the configuration file instead of being searched "
    "for. And the numerical path has a calculator: the model writes a short program for the "
    "arithmetic, the program is run in an isolated interpreter, and the answer is built on "
    "the values it prints.", body))

add(Paragraph("Why the adapter is switched rather than always on", h2))
add(Paragraph(
    "Fine-tuning helped and hurt at the same time. On identical held-back questions and "
    "passages, the adapter took calculations from 0.74 to 0.47, while taking citation of the "
    "supplied passage from 55% to 79%, refusal of other courses' questions from 38% to 100%, "
    "and cutting the assessed work handed over from 32 of 33 solvable problems to 3 of 20. It "
    "made no difference to true-or-false statements about the notes. Applying it at part "
    "strength did not recover the arithmetic: at a quarter strength the calculations still "
    "scored 0.50, and the answers still claimed that a 6 kHz tone sampled at 8 kHz does not "
    "alias. Switching it per question keeps both halves of that result, and costs nothing "
    "measurable, because one model is loaded either way.", body))

add(Paragraph("3. How it is tested", h1))
add(Paragraph(
    "Testing has two parts. Most of it marks itself, so it can be re-run after every change "
    "at no cost beyond GPU time. A small part is marked by hand, because no rule can judge "
    "whether an explanation is any good.", body))

add(Paragraph("Tests that mark themselves", h2))
add(table([
    ["Test", "Size", "What it measures"],
    ["Calculations", "264", "Generated problems across eleven kinds, each with its answer "
                            "computed in Python, in three different phrasings"],
    ["Concepts", "200 statements", "True-or-false pairs from the notes, each with the sentence "
                                   "that settles it; a pair differs in one edit, so answering "
                                   "'true' to everything scores exactly half"],
    ["Course boundary", "160", "Course questions against questions from other courses and "
                               "other faculties, asked two ways each"],
    ["Assessed work", "54 conversations", "A problem framed as homework due for marks, then two "
                                          "more turns of pressure"],
    ["Follow-ups", "54 conversations", "A calculation, then the same problem with one number "
                                       "changed"],
    ["Error finding", "72", "A student's working, three quarters of it containing a classic "
                            "mistake and a quarter correct"],
    ["Course questions", "23", "Facts from the 2026 course handout"],
    ["Speed, cost, memory", "every answer", "Seconds per answer, API charge, and whether the "
                                            "card ran out of memory"],
], [3.3 * cm, 2.5 * cm, W - 5.8 * cm]))

add(Paragraph("The part marked by hand", h2))
add(Paragraph(
    "Forty-seven benchmark questions, written before any model was trained, are answered by "
    "three systems: this tutor, the same pipeline without fine-tuning, and Claude Opus 5 with "
    "the course prompt. The 141 answers are pooled per question, shuffled and labelled A, B "
    "and C, so the marker cannot tell which system wrote which. Each answer is marked point by "
    "point against a marking guide fixed in advance, and for whether it states anything the "
    "course material does not support.", body))
add(Paragraph(
    "One marker is a weakness, so two checks sit around it. The hand marks are compared with "
    "the automatic rules on the twelve calculation questions, which says whether the automatic "
    "marking can be trusted; and twenty answers are re-marked blind at least three days later, "
    "which says whether the marking is self-consistent.", body))
add(Paragraph(
    "The second check has since been run, and it failed: 14 of the 20 answers kept their label "
    "against a target of 17. Splitting it shows where the trouble is. On the points written "
    "into the marking guide the two markings agree 17 times out of 20 and the mean fraction of "
    "points moved by 0.020. On whether an answer goes beyond the course material they agree far "
    "less: the same twenty answers went from 5 judged unfaithful to 10, every change in the "
    "same direction. A point is a written claim that can be checked; faithfulness asks a marker "
    "to weigh an answer against 109 files nobody holds in mind, with the passages not in front "
    "of them, so the judgement rests on recollection and moves between sittings. The figures "
    "for correctness stand; the figure for faithfulness is reported with this beside it.", body))

add(Paragraph("What keeps the result honest", h2))
add(table([
    ["Written down first",
     "Twenty-one acceptance tests, each with a pass mark and a reason, plus three checks on the "
     "evaluation itself, were recorded before any result was read. A missed mark is reported as "
     "a miss, and the one test withdrawn is reported as withdrawn."],
    ["Held-back questions",
     "Every test has a development half, used while improving the tutor, and a half it has "
     "never seen. The acceptance tests are decided on the half it has never seen."],
    ["Frozen system",
     "The tutor was frozen on 21 September, before the final test was generated or run."],
    ["Statistics, not single numbers",
     "Every rate carries a 95% confidence interval; the fine-tuned and untuned systems are "
     "compared question by question with a bootstrap interval and an exact test."],
    ["Faults recorded, not buried",
     "Two flaws found in my own tests are written into the protocol: that the first scope test "
     "reused subject areas it should have avoided, and that one calculation rule ignored the "
     "sign of a pole. The final test also exposed a routing fault in the frozen system, which "
     "sent 125 of 264 calculations to the wrong weights; it was fixed, everything affected was "
     "run again, and both runs are kept."],
], [4.2 * cm, W - 4.2 * cm], header=False))

add(KeepTogether([Paragraph("4. Where it stands", h1), Paragraph(
    "The final test is complete. The table compares, on identical held-back questions, the "
    "fine-tuned model alone, the same model with no fine-tuning, and the system as frozen. "
    "Of the criteria set before any result was read, twenty-two could be decided and the "
    "frozen system passes eight; one was withdrawn from scope.", body)]))

add(table([
    ["Measure", "Fine-tuned", "Untuned", "As frozen"],
    ["Calculations, marking points", "0.47", "0.74", "0.74"],
    ["Calculations fully right", "84 of 263", "156 of 264", "155 of 263"],
    ["True-or-false on the notes", "92 of 101", "95 of 101", "92 of 101"],
    ["Cites the passage it was given", "68 of 86", "47 of 86", "68 of 86"],
    ["Declines other courses", "80 of 80", "30 of 80", "80 of 80"],
    ["Wrongly refuses its own course", "3 of 78", "0 of 80", "2 of 78"],
    ["Gives assessed work away, when solvable", "3 of 20", "32 of 33", "5 of 33"],
], [6.2 * cm, (W - 6.2 * cm) / 3, (W - 6.2 * cm) / 3, (W - 6.2 * cm) / 3], align_right=(1, 2, 3)))

add(Spacer(1, 8))
add(Paragraph("After the final test", h2))
add(Paragraph(
    "Two changes, each prompted by what the test found, were measured afterwards on the same "
    "held-back questions. They do not replace the acceptance results, because a system changed "
    "after seeing its result has not been tested blind.", body))
add(table([
    ["Measure", "As tested", "After"],
    ["Calculations, marking points", "0.735", "0.816"],
    ["Corrects a student's mistake", "34 of 54", "38 of 53"],
    ["Questions about how the course runs", "12 of 23", "22 of 23"],
    ["Follow-up questions right", "38 of 54", "33 of 53"],
], [8.2 * cm, (W - 8.2 * cm) / 2, (W - 8.2 * cm) / 2], align_right=(1, 2)))
add(Paragraph(
    "The calculator's gain on calculations is +0.081, with a 95% interval of +0.038 to "
    "+0.125. Its cost is the last row: follow-up questions got worse, and answers take about "
    "three seconds longer.", body))

add(Spacer(1, 10))
add(Paragraph(
    "Everything in this document is reproducible from the repository: the test banks and their "
    "fingerprints, the protocols with their dated amendments, every run's answers, and the "
    "code that scored them.", note))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2.2 * cm,
                        title="SignalsBot overview", author="Sanjan Naidoo")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
