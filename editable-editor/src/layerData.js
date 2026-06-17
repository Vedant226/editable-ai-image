export async function getLayerData() {
  try {
    const response =
      await fetch(
        "/layers/metadata.json"
      );

    const metadata =
      await response.json();

    return metadata.map(
      (
        layer,
        index
      ) => ({
        id:
          layer.id ??
          index,

        file:
          layer.file,

        type:
          (
            layer.type ||
            ""
          ).toLowerCase(),

        text:
          layer.text ||
          "",

        x:
          Number(
            layer.x
          ) || 0,

        y:
          Number(
            layer.y
          ) || 0,

        width:
          Number(
            layer.width
          ) || 100,

        height:
          Number(
            layer.height
          ) || 50,

        rotation:
          Number(
            layer.rotation
          ) || 0,

        zIndex:
          Number(
            layer.zIndex
          ) || 0,

        fontFamily:
          layer.fontFamily ||
          "Cinzel",

        fontWeight:
          layer.fontWeight ||
          "bold",

        fontColor:
          layer.fontColor ||
          "#d8b36a",

        strokeColor:
          layer.strokeColor ||
          "#5a2e12",

        strokeWidth:
          Number(
            layer.strokeWidth
          ) || 1.5,

        letterSpacing:
          Number(
            layer.letterSpacing
          ) || 1,

        textAlign:
          layer.textAlign ||
          "center",

        shadowBlur:
          Number(
            layer.shadowBlur
          ) || 2,

        visible: true,

        edited: false
      })
    );
  } catch (error) {
    console.error(
      "Failed loading metadata:",
      error
    );

    return [];
  }
}