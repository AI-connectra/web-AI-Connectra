# Website images

Place optimized public images in this folder. Recommended formats are WebP or AVIF.

Use them from Razor markup with root-relative paths:

```html
<img src="~/images/bridge-crack-detection.webp"
     alt="AI bridge crack detection example"
     loading="lazy"
     width="1200"
     height="800">
```

Keep private or user-uploaded files outside `wwwroot` and serve them through an authorized endpoint instead.
