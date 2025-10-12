/**
 * ARC color palette and utilities
 */

export const ARC_COLORS = [
  '#000000', // 0: black
  '#0074D9', // 1: blue
  '#FF4136', // 2: red
  '#2ECC40', // 3: green
  '#FFDC00', // 4: yellow
  '#AAAAAA', // 5: grey
  '#F012BE', // 6: magenta
  '#FF851B', // 7: orange
  '#7FDBFF', // 8: light blue
  '#870C25', // 9: maroon
  '#FFFFFF', // 10: white
];

export const ARC_COLOR_NAMES = [
  'black',
  'blue',
  'red',
  'green',
  'yellow',
  'grey',
  'magenta',
  'orange',
  'light blue',
  'maroon',
  'white',
];

export function getColorCode(color: number): string {
  if (color < 0 || color >= ARC_COLORS.length) {
    return ARC_COLORS[0]; // Default to black
  }
  return ARC_COLORS[color];
}

export function getColorName(color: number): string {
  if (color < 0 || color >= ARC_COLOR_NAMES.length) {
    return 'unknown';
  }
  return ARC_COLOR_NAMES[color];
}
