import streamlit as st
import os

def load_css():
    """
    Charge le fichier style.css et l'injecte dans la page Streamlit pour appliquer
    l'identité visuelle de SASRec Studio.
    """
    if os.path.exists('style.css'):
        with open('style.css', 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def score_badge(score):
    """
    Retourne un badge HTML coloré selon le niveau du score de recommandation.
    - Élevé (>= 0.8) : Vert
    - Moyen (0.5 - 0.8) : Orange/Marron
    - Faible (< 0.5) : Rouge
    """
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        return '<span class="score-badge score-low">N/A</span>'
        
    if score_val >= 0.8:
        return f'<span class="score-badge score-high">Élevé ({score_val:.2f})</span>'
    elif score_val >= 0.5:
        return f'<span class="score-badge score-medium">Moyen ({score_val:.2f})</span>'
    else:
        return f'<span class="score-badge score-low">Faible ({score_val:.2f})</span>'

def metric_card(icon, label, value, delta=None):
    """
    Affiche une carte de métriques (Metric Card) responsive et stylisée
    avec une icône Tabler à gauche, un label et sa valeur associée.
    """
    if icon and not icon.startswith('ti '):
        icon = f"ti {icon}"
        
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            prefix = "+" if delta >= 0 else ""
            klass = "positive" if delta >= 0 else "negative"
            icon_delta = "ti-arrow-up-right" if delta >= 0 else "ti-arrow-down-right"
            delta_html = f'<div class="metric-delta {klass}"><i class="ti {icon_delta}"></i> {prefix}{delta}</div>'
        else:
            delta_html = f'<div class="metric-delta">{delta}</div>'
            
    html = f"""
    <div class="metric-card">
        <div class="metric-icon-container">
            <i class="{icon}"></i>
        </div>
        <div class="metric-details">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def product_card(rank, name, image_url, score, price, category):
    """
    Génère le code HTML d'une carte de produit (Product Card) pour les recommandations.
    Retourne la chaîne HTML. Doit être affichée dans un container avec la classe 'product-grid'.
    """
    try:
        score_val = float(score)
        score_pct = min(max(int(score_val * 100), 0), 100)
    except (ValueError, TypeError):
        score_val = 0.0
        score_pct = 0
        
    badge_html = score_badge(score_val)
    
    # Image par défaut si aucune URL n'est fournie
    if not image_url:
        image_html = '<div style="font-size: 64px; color: #85B7EB; display: flex; align-items: center; justify-content: center; height: 100%;"><i class="ti ti-package"></i></div>'
    else:
        image_html = f'<img src="{image_url}" alt="{name}">'
        
    html = f"""
    <div class="product-card">
        <div class="product-image-container">
            {image_html}
            <div class="product-rank-badge">#{rank}</div>
            <div class="product-category-tag">{category}</div>
        </div>
        <div class="product-card-body">
            <div class="product-name" title="{name}">{name}</div>
            <div class="product-score-section">
                <div class="product-score-label">Score: {score_val:.5f}</div>
                {badge_html}
            </div>
            <div class="product-score-bar-bg">
                <div class="product-score-bar-fill" style="width: {score_pct}%;"></div>
            </div>
            <div class="product-price">{price}</div>
        </div>
    </div>
    """
    return html

def product_cards_grid(cards_html_list):
    """
    Affiche une liste de Product Cards au sein d'une grille HTML responsive (3 colonnes).
    """
    grid_content = "".join(cards_html_list)
    st.markdown(f'<div class="product-grid">{grid_content}</div>', unsafe_allow_html=True)

def section_header(title, subtitle, icon):
    """
    Affiche un bandeau d'en-tête (Header) pleine largeur de page stylisé pour SASRec Studio.
    """
    if icon and not icon.startswith('ti '):
        icon = f"ti {icon}"
        
    html = f"""
    <div class="page-header">
        <div class="page-header-content">
            <div class="page-header-icon">
                <i class="{icon}"></i>
            </div>
            <div>
                <div class="page-header-title">{title}</div>
                <div class="page-header-subtitle">{subtitle}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def model_status_badge(is_loaded):
    """
    Retourne et affiche un badge d'état du modèle chargé dans la sidebar.
    """
    if is_loaded:
        html = """
        <div class="sidebar-status status-loaded">
            <span class="status-pulse loaded" style="background-color: #3B6D11;"></span>Modèle Chargé
        </div>
        """
    else:
        html = """
        <div class="sidebar-status status-unloaded">
            <span class="status-pulse unloaded" style="background-color: #854F0B;"></span>Non Entraîné
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def sidebar_logo():
    """
    Affiche le logo personnalisé "SASRec Studio" et sa tagline dans la sidebar.
    """
    html = """
    <div class="sidebar-logo" style="display: flex; align-items: center; gap: 8px; font-size: 20px; font-weight: 500; color: #042C53; margin-bottom: 2px;">
        <i class="ti ti-brain" style="font-size: 24px; color: #378ADD;"></i>
        <span>SASRec Studio</span>
    </div>
    <div class="sidebar-tagline" style="font-size: 12px; color: #888780; margin-bottom: 1.5rem;">
        Sequential Recommendation
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
