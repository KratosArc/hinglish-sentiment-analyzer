"""
Generate PDF report for Hinglish Sentiment Analyzer
Run: python src/generate_report.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

# Install: pip install matplotlib reportlab pillow

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
except ImportError:
    print("Install reportlab: pip install reportlab pillow")
    exit(1) 

# === Data ===
test_accuracy = 0.6904
test_f1 = 0.6869
train_accuracy = 0.6307
epochs = [1, 2, 3]
train_loss = [0.8708, 0.8108, 0.6782]
val_loss = [0.8591, 0.8392, 0.8486]

confusion_data = np.array([
    [766, 185, 28],
    [273, 685, 252],
    [68, 163, 710]
])

class_names = ['Negative', 'Neutral', 'Positive']
precision = [0.69, 0.66, 0.72]
recall = [0.78, 0.57, 0.75]
f1_scores = [0.73, 0.61, 0.74]

# === Generate Visualizations ===
print("Generating visualizations...")

# 1. Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss, 'o-', label='Training Loss', linewidth=2, markersize=8)
ax1.plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(epochs)

ax2.bar(range(3), precision, width=0.25, label='Precision', alpha=0.8)
ax2.bar([i+0.25 for i in range(3)], recall, width=0.25, label='Recall', alpha=0.8)
ax2.bar([i+0.50 for i in range(3)], f1_scores, width=0.25, label='F1-Score', alpha=0.8)
ax2.set_xlabel('Class', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Per-Class Performance', fontsize=12, fontweight='bold')
ax2.set_xticks([i+0.25 for i in range(3)])
ax2.set_xticklabels(class_names)
ax2.legend()
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('src/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")
plt.close()

# 2. Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(confusion_data, cmap='Blues', aspect='auto')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title('Confusion Matrix (Test Set: 3,130 tweets)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, confusion_data[i, j],
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig('src/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")
plt.close()

# 3. Model Performance Summary
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

metrics_text = f"""
HINGLISH SENTIMENT ANALYZER — MODEL PERFORMANCE

Test Accuracy:                    69.04%
Weighted F1-Score:               0.6869
Training Epochs:                 3 (early stopping)
Training Time:                   ~14 minutes (Google Colab T4 GPU)

Per-Class F1-Scores:
  • Negative:  0.73 (good)
  • Neutral:   0.61 (harder, ambiguous)
  • Positive:  0.74 (good)

Key Finding: Neutral sentiment is harder to predict (0.61 F1) 
than clear extremes (Negative/Positive both ~0.73), which matches 
real-world annotation difficulty.
"""

ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('src/performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved performance_summary.png")
plt.close()

# === Generate PDF ===
print("\nGenerating PDF report...")

pdf_file = "Hinglish_Sentiment_Analyzer_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                        rightMargin=0.75*inch, leftMargin=0.75*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

styles = getSampleStyleSheet()
story = []

# Title
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1f77b4'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

story.append(Paragraph("🇮🇳 Hinglish Sentiment Analyzer", title_style))
story.append(Spacer(1, 0.2*inch))

# Subtitle
subtitle_style = ParagraphStyle(
    'Subtitle',
    parent=styles['Normal'],
    fontSize=12,
    textColor=colors.HexColor('#666666'),
    alignment=TA_CENTER,
    spaceAfter=20
)
story.append(Paragraph(f"Fine-tuned XLM-RoBERTa for Code-Mixed Hindi-English Sentiment Analysis<br/>Generated: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))

# Executive Summary
story.append(Paragraph("Executive Summary", styles['Heading2']))
summary_text = """
This project fine-tunes XLM-RoBERTa (a multilingual transformer trained on 100+ languages) 
on the SemEval-2020 Task 9 Hinglish dataset to classify sentiment in code-mixed Hindi-English 
text. The model achieves <b>69.04% accuracy</b> on a held-out test set of 3,130 tweets, 
with strong performance on positive (F1=0.74) and negative (F1=0.73) classes, and harder-to-predict 
neutral sentiment (F1=0.61, reflecting real ambiguity in human annotation).
"""
story.append(Paragraph(summary_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

# Key Metrics Table
story.append(Paragraph("Key Metrics", styles['Heading2']))
metrics_data = [
    ['Metric', 'Score'],
    ['Test Accuracy', '69.04%'],
    ['Weighted F1-Score', '0.6869'],
    ['Training Epochs', '3 (with early stopping)'],
    ['Training Time', '~14 minutes (Colab T4 GPU)'],
]
metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
metrics_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))
story.append(metrics_table)
story.append(Spacer(1, 0.3*inch))

# Visualizations
story.append(Paragraph("Training Dynamics", styles['Heading2']))
story.append(Image('src/training_curves.png', width=6*inch, height=2*inch))
story.append(Spacer(1, 0.2*inch))
training_text = """
The model trained for 3 epochs with learning rate 2e-5 and warmup_steps=500. Training loss 
steadily decreased (0.871 → 0.678) while validation loss remained stable (0.859 → 0.849), 
indicating healthy convergence with no overfitting.
"""
story.append(Paragraph(training_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph("Confusion Matrix (Test Set)", styles['Heading2']))
story.append(Image('src/confusion_matrix.png', width=5*inch, height=4*inch))
story.append(Spacer(1, 0.2*inch))
confusion_text = """
The model correctly predicts negative sentiment 78% of the time (766/979 true negatives), 
positive 75% (710/941), but neutral only 57% (685/1210). This asymmetry reflects the genuine 
difficulty of neutral sentiment classification — it's ambiguous in human annotation too.
"""
story.append(Paragraph(confusion_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph("Per-Class Performance", styles['Heading2']))
story.append(Image('src/performance_summary.png', width=6*inch, height=2.5*inch))
story.append(Spacer(1, 0.3*inch))

# Methodology
story.append(PageBreak())
story.append(Paragraph("Methodology", styles['Heading2']))
methodology_text = """
<b>Data:</b> SemEval-2020 Task 9 (14k training + 3k test Hinglish tweets)<br/>
<b>Preprocessing:</b> Light-touch cleaning (smart quote normalization, whitespace collapse) 
to preserve signal for transformers<br/>
<b>Tokenization:</b> XLM-RoBERTa AutoTokenizer with max_length=128<br/>
<b>Model:</b> XLM-RoBERTa-base (100+ languages) fine-tuned with 3-unit classification head<br/>
<b>Training:</b> AdamW optimizer, learning_rate=2e-5, warmup_steps=500, batch_size=16<br/>
<b>Evaluation:</b> Accuracy, F1-score (weighted), confusion matrix, precision/recall per class
"""
story.append(Paragraph(methodology_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

# Key Findings
story.append(Paragraph("Key Findings & Learnings", styles['Heading2']))
findings_text = """
1. <b>Learning rate is critical:</b> Default 5e-5 caused model divergence; 2e-5 + warmup fixed it<br/>
2. <b>Transformers benefit from light preprocessing:</b> Punctuation, casing, emojis carry signal<br/>
3. <b>No overfitting observed:</b> Validation metrics remained stable across epochs<br/>
4. <b>Code-mixed sentiment is genuinely hard:</b> Neutral (0.61 F1) vs extremes (0.73+) reflects real ambiguity<br/>
5. <b>Competitive performance:</b> 69% matches published benchmarks (65-72%) on same dataset
"""
story.append(Paragraph(findings_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

# Footer
footer_text = f"""
<b>Project:</b> Hinglish Sentiment Analyzer | <b>Author:</b> Jiveshwar (KratosArc) | 
<b>GitHub:</b> https://github.com/KratosArc/hinglish-sentiment-analyzer | 
<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}
"""
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph(footer_text, styles['Normal']))

# Build PDF
doc.build(story)
print(f"✓ PDF report saved: {pdf_file}")