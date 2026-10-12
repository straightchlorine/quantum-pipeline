"""Make the home page og:title and twitter:title match its <title>.

Material's social plugin uses only the site name for the home page, while
<title> is "<front matter title> - <site name>".
"""


def on_post_page(output, page, config):
    title = page.meta.get("title")
    if page.is_homepage and title:
        full = f"{title} - {config.site_name}"
        for prop in ("og:title", "twitter:title"):
            output = output.replace(
                f'<meta property="{prop}" content="{config.site_name}" />',
                f'<meta property="{prop}" content="{full}" />',
            )
    output = output.replace('<meta property="twitter:', '<meta name="twitter:')
    for cls, label in (("info", "Image information"), ("reset", "Reset zoom")):
        output = output.replace(
            f'<button class="panzoom-{cls} panzoom-button">',
            f'<button class="panzoom-{cls} panzoom-button" aria-label="{label}">',
        )
    return output
