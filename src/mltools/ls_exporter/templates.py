"""UI templates and constants for Label Studio project generation."""

LABEL_COLORS = [
    "green", "blue", "red", "orange", "purple",
    "cyan", "magenta", "yellow", "brown", "pink",
]

POLYGON_XML_LAYOUT = """\
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="mask" toName="image">
{label_tags}  </PolygonLabels>
  <Number name="score" toName="image" perRegion="true" editable="false"/>
</View>"""