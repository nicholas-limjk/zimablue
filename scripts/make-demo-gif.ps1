$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outDir = Join-Path $root ("artifacts\linkedin-demo-frames-" + [DateTimeOffset]::Now.ToUnixTimeSeconds())
$gifPath = Join-Path $root "artifacts\zima-blue-linkedin-demo.gif"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$width = 960
$height = 540
$cols = 22
$rows = 12
$cell = 22
$gap = 3
$ox = 38
$oy = 118

function X($gridX) { return $script:ox + ($gridX * ($script:cell + $script:gap)) }
function Y($gridY) { return $script:oy + ($gridY * ($script:cell + $script:gap)) }

$rocks = @{}
function Add-Wall($x, $fromY, $toY, $openingY) {
  for ($y = $fromY; $y -le $toY; $y += 1) {
    if ($y -ne $openingY) {
      $script:rocks["$x,$y"] = $true
    }
  }
}
Add-Wall 6 0 11 5
Add-Wall 16 0 11 6
@(
  @(3,2), @(3,3), @(3,7), @(3,8), @(8,1), @(9,1), @(10,1), @(11,1),
  @(9,2), @(9,3), @(9,9), @(10,9), @(11,9), @(12,9), @(13,9), @(14,9),
  @(14,2), @(14,3), @(12,10), @(13,10)
) | ForEach-Object { $rocks["$($_[0]),$($_[1])"] = $true }

function In-Wind1($x, $y) { return $x -ge 8 -and $x -lt 16 -and $y -ge 3 -and $y -lt 9 }
function In-Wind2($x, $y) { return $x -ge 17 -and $x -lt 21 -and $y -ge 4 -and $y -lt 9 }

$coldPath = @(
  @(1,1), @(2,1), @(3,1), @(4,1), @(5,1), @(5,2), @(5,3), @(5,4), @(5,5), @(6,5),
  @(5,5), @(4,6), @(3,7), @(2,8), @(2,9), @(2,10), @(3,10), @(4,10), @(5,10),
  @(5,9), @(5,8), @(5,7), @(5,6), @(5,5), @(6,5), @(7,5), @(8,5), @(9,6),
  @(10,5), @(11,6), @(12,5), @(13,6), @(14,6), @(15,6), @(14,6), @(13,6),
  @(12,6), @(12,5), @(12,4), @(12,5), @(12,6), @(12,7), @(12,8), @(13,8),
  @(14,7), @(15,6), @(16,6)
)
$warmPath = @(
  @(1,1), @(2,1), @(2,2), @(2,3), @(2,4), @(2,5), @(2,6), @(2,7), @(2,8),
  @(2,9), @(2,10), @(3,10), @(4,9), @(5,8), @(5,7), @(5,6), @(5,5), @(6,5),
  @(7,5), @(8,5), @(9,5), @(10,5), @(11,5), @(12,5), @(12,4), @(12,5),
  @(12,6), @(12,7), @(12,8), @(13,8), @(14,7), @(15,6), @(16,6), @(17,6),
  @(18,6), @(19,6), @(20,6), @(21,6)
)

function Convert-PathPoints($items) {
  if ($items.Count -gt 0 -and $items[0] -is [array]) {
    return @($items | ForEach-Object { [pscustomobject]@{ x = [double]$_[0]; y = [double]$_[1] } })
  }

  $points = @()
  for ($i = 0; $i -lt $items.Count; $i += 2) {
    $points += [pscustomobject]@{ x = [double]$items[$i]; y = [double]$items[$i + 1] }
  }
  return $points
}

$coldPath = Convert-PathPoints $coldPath
$warmPath = Convert-PathPoints $warmPath

$phases = @(
  @{ t = 0.0; title = "Base impulse only"; caption = "The actor can only move east/south and ask: target reached?"; badges = @() },
  @{ t = 1.3; title = "Round 1: invent route search"; caption = "The thinking loop writes a route-search skill after simple motion is not enough."; badges = @("route search") },
  @{ t = 2.7; title = "Locked gate discovered"; caption = "The route hits a lock. Only now does key search become useful."; badges = @("route search", "key search") },
  @{ t = 4.0; title = "Unstable wind region"; caption = "Moves drift in the wind, so the agent trains a controller."; badges = @("key", "controller") },
  @{ t = 5.4; title = "Puzzle clue at the seal"; caption = "A failed switch probe makes the clue relevant: prime plates, make forty."; badges = @("controller", "read clue") },
  @{ t = 6.8; title = "Writes switch procedure"; caption = "The solver selects the large primes 17 and 23 from the clue."; badges = @("17 + 23 = 40") },
  @{ t = 8.0; title = "Round 1 fails late"; caption = "The seal opens, but the strict clock expires before the target."; badges = @("learned skills persist") },
  @{ t = 9.4; title = "Round 2: remembered skills"; caption = "The next run starts with persisted skills and clears the world."; badges = @("skills: 6", "warm start") },
  @{ t = 11.0; title = "Target reached"; caption = "Base impulse falls to 0. The actor rests."; badges = @("complete") }
)

function Lerp($a, $b, $p) { return $a + (($b - $a) * $p) }
function Smooth($p) { return $p * $p * (3 - (2 * $p)) }

function Get-Agent($t) {
  if ($t -lt 8.4) {
    $path = $script:coldPath
    $local = $t / 8.4
  } else {
    $path = $script:warmPath
    $local = ($t - 8.4) / 4.1
  }
  $n = $path.Count - 1
  $raw = [Math]::Max(0, [Math]::Min($n - 0.001, $local * $n))
  $index = [Math]::Floor($raw)
  $p = Smooth ($raw - $index)
  $a = $path[$index]
  $b = $path[$index + 1]
  return @( (Lerp $a.x $b.x $p), (Lerp $a.y $b.y $p) )
}

function Get-Phase($t) {
  $current = $script:phases[0]
  foreach ($phase in $script:phases) {
    if ($t -ge $phase.t) {
      $current = $phase
    }
  }
  return $current
}

function Draw-RoundedRect($graphics, $brush, $pen, $x, $y, $w, $h, $r) {
  $path = [System.Drawing.Drawing2D.GraphicsPath]::new()
  $d = $r * 2
  $path.AddArc($x, $y, $d, $d, 180, 90)
  $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
  $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
  $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
  $path.CloseFigure()
  if ($brush) { $graphics.FillPath($brush, $path) }
  if ($pen) { $graphics.DrawPath($pen, $path) }
  $path.Dispose()
}

function Draw-Cell($graphics, $x, $y, $fill, $text, $color) {
  $brush = [System.Drawing.SolidBrush]::new($fill)
  $pen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(40, 25, 33, 42), 1)
  Draw-RoundedRect $graphics $brush $pen (X $x) (Y $y) $script:cell $script:cell 4
  $brush.Dispose()
  $pen.Dispose()
  if ($text) {
    $font = [System.Drawing.Font]::new("Segoe UI", 8, [System.Drawing.FontStyle]::Bold)
    $textBrush = [System.Drawing.SolidBrush]::new($color)
    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $rect = [System.Drawing.RectangleF]::new((X $x), (Y $y), $script:cell, $script:cell)
    $graphics.DrawString($text, $font, $textBrush, $rect, $format)
    $format.Dispose()
    $textBrush.Dispose()
    $font.Dispose()
  }
}

function Draw-Wrapped($graphics, $text, $font, $brush, $x, $y, $width, $lineHeight) {
  $line = ""
  foreach ($word in $text.Split(" ")) {
    $test = if ($line) { "$line $word" } else { $word }
    if ($graphics.MeasureString($test, $font).Width -gt $width) {
      $graphics.DrawString($line, $font, $brush, [single]$x, [single]$y)
      $y += $lineHeight
      $line = $word
    } else {
      $line = $test
    }
  }
  if ($line) { $graphics.DrawString($line, $font, $brush, [single]$x, [single]$y) }
  return $y + $lineHeight
}

$fonts = @{
  title = [System.Drawing.Font]::new("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)
  small = [System.Drawing.Font]::new("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
  phase = [System.Drawing.Font]::new("Segoe UI", 18, [System.Drawing.FontStyle]::Bold)
  caption = [System.Drawing.Font]::new("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
  badge = [System.Drawing.Font]::new("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
  agent = [System.Drawing.Font]::new("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
}

$colors = @{
  ink = [System.Drawing.Color]::FromArgb(25, 33, 42)
  muted = [System.Drawing.Color]::FromArgb(98, 113, 124)
  paper = [System.Drawing.Color]::FromArgb(247, 244, 236)
  panel = [System.Drawing.Color]::FromArgb(255, 253, 247)
  line = [System.Drawing.Color]::FromArgb(216, 209, 194)
}

$frames = 90
$duration = 12.5

for ($i = 0; $i -lt $frames; $i += 1) {
  $t = ($i / ($frames - 1)) * $duration
  $bitmap = [System.Drawing.Bitmap]::new($width, $height)
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $graphics.Clear($colors.paper)

  $inkBrush = [System.Drawing.SolidBrush]::new($colors.ink)
  $mutedBrush = [System.Drawing.SolidBrush]::new($colors.muted)
  $graphics.DrawString("Zima Blue toy agent: skills emerge from failure", $fonts.title, $inkBrush, 34, 34)
  $graphics.DrawString("base impulse: reach target", $fonts.small, $mutedBrush, 36, 72)

  $panelBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(236, 255, 253, 247))
  $linePen = [System.Drawing.Pen]::new($colors.line, 1)
  Draw-RoundedRect $graphics $panelBrush $linePen 24 100 640 390 8
  $panelBrush.Dispose()
  $linePen.Dispose()

  for ($y = 0; $y -lt $rows; $y += 1) {
    for ($x = 0; $x -lt $cols; $x += 1) {
      $key = "$x,$y"
      if ($rocks.ContainsKey($key)) {
        Draw-Cell $graphics $x $y ([System.Drawing.Color]::FromArgb(89, 100, 106)) "" $colors.ink
      } elseif (In-Wind1 $x $y) {
        Draw-Cell $graphics $x $y ([System.Drawing.Color]::FromArgb(128, 108, 173)) ">" ([System.Drawing.Color]::White)
      } elseif (In-Wind2 $x $y) {
        Draw-Cell $graphics $x $y ([System.Drawing.Color]::FromArgb(189, 75, 95)) "v" ([System.Drawing.Color]::White)
      } else {
        Draw-Cell $graphics $x $y ([System.Drawing.Color]::FromArgb(244, 255, 253, 247)) "" $colors.ink
      }
    }
  }

  Draw-Cell $graphics 2 10 ([System.Drawing.Color]::FromArgb(242, 199, 106)) "K" ([System.Drawing.Color]::FromArgb(91, 66, 20))
  Draw-Cell $graphics 6 5 ([System.Drawing.Color]::FromArgb(106, 75, 51)) "G" ([System.Drawing.Color]::White)
  Draw-Cell $graphics 15 6 ([System.Drawing.Color]::FromArgb(255, 241, 173)) "?" $colors.ink
  Draw-Cell $graphics 16 6 ([System.Drawing.Color]::FromArgb(21, 25, 35)) "LOCK" ([System.Drawing.Color]::White
  )
  Draw-Cell $graphics 21 6 ([System.Drawing.Color]::FromArgb(216, 79, 63)) "*" ([System.Drawing.Color]::White)
  Draw-Cell $graphics 12 4 ([System.Drawing.Color]::FromArgb(39, 85, 78)) "17" ([System.Drawing.Color]::White)
  Draw-Cell $graphics 12 8 ([System.Drawing.Color]::FromArgb(39, 85, 78)) "23" ([System.Drawing.Color]::White)
  Draw-Cell $graphics 11 10 ([System.Drawing.Color]::FromArgb(112, 79, 87)) "21" ([System.Drawing.Color]::White)

  $agent = Get-Agent $t
  $ax = (X $agent[0]) + ($cell / 2)
  $ay = (Y $agent[1]) + ($cell / 2)
  $agentBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(47, 136, 200))
  $whitePen = [System.Drawing.Pen]::new([System.Drawing.Color]::White, 3)
  $graphics.FillEllipse($agentBrush, [single]($ax - 13), [single]($ay - 13), 26, 26)
  $graphics.DrawEllipse($whitePen, [single]($ax - 13), [single]($ay - 13), 26, 26)
  $format = [System.Drawing.StringFormat]::new()
  $format.Alignment = [System.Drawing.StringAlignment]::Center
  $format.LineAlignment = [System.Drawing.StringAlignment]::Center
  $graphics.DrawString("A", $fonts.agent, [System.Drawing.Brushes]::White, [System.Drawing.RectangleF]::new([single]($ax - 13), [single]($ay - 13), 26, 26), $format)
  $format.Dispose()
  $agentBrush.Dispose()
  $whitePen.Dispose()

  $sideBrush = [System.Drawing.SolidBrush]::new($colors.panel)
  $sidePen = [System.Drawing.Pen]::new($colors.line, 1)
  Draw-RoundedRect $graphics $sideBrush $sidePen 694 100 226 390 8
  $sideBrush.Dispose()
  $sidePen.Dispose()
  $phase = Get-Phase $t
  $yText = Draw-Wrapped $graphics $phase.title $fonts.phase $inkBrush 716 144 180 24
  $yText += 12
  $yText = Draw-Wrapped $graphics $phase.caption $fonts.caption $mutedBrush 716 $yText 176 20
  $yText += 18
  foreach ($badge in $phase.badges) {
    $badgeBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(236, 247, 239))
    $badgePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(155, 201, 165), 1)
    Draw-RoundedRect $graphics $badgeBrush $badgePen 716 $yText 166 26 13
    $badgeBrush.Dispose()
    $badgePen.Dispose()
    $greenBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(47, 127, 95))
    $graphics.DrawString($badge, $fonts.badge, $greenBrush, 728, ($yText + 5))
    $greenBrush.Dispose()
    $yText += 34
  }
  $progress = $t / $duration
  $trackBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(235, 227, 213))
  $progressBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(47, 136, 200))
  Draw-RoundedRect $graphics $trackBrush $null 716 448 166 10 5
  Draw-RoundedRect $graphics $progressBrush $null 716 448 ([int](166 * $progress)) 10 5
  $trackBrush.Dispose()
  $progressBrush.Dispose()

  $inkBrush.Dispose()
  $mutedBrush.Dispose()
  $graphics.Dispose()
  $bitmap.Save((Join-Path $outDir ("frame-{0:D3}.png" -f $i)), [System.Drawing.Imaging.ImageFormat]::Png)
  $bitmap.Dispose()
}

$encoder = [System.Windows.Media.Imaging.GifBitmapEncoder]::new()
Get-ChildItem -LiteralPath $outDir -Filter "frame-*.png" | Sort-Object Name | ForEach-Object {
  $stream = [System.IO.File]::OpenRead($_.FullName)
  try {
    $frame = [System.Windows.Media.Imaging.BitmapFrame]::Create($stream, [System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat, [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
    $encoder.Frames.Add($frame)
  } finally {
    $stream.Dispose()
  }
}

$gifStream = [System.IO.File]::Create($gifPath)
try {
  $encoder.Save($gifStream)
} finally {
  $gifStream.Dispose()
}

foreach ($font in $fonts.Values) {
  $font.Dispose()
}

Write-Output $gifPath
