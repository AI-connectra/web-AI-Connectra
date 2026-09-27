# AI Connectra — ASP.NET Core Razor Pages

This project preserves the existing AI Connectra design and client-side behavior, but hosts it as an ASP.NET Core Razor Pages application. The catalog contains 200 capstone projects across 10 disciplines. Every project uses the same data-driven detail view, so adding a project means adding data rather than creating another Razor page.

## Open in Visual Studio

1. Open `AI.Connectra.sln` in Visual Studio 2022 or later.
2. Select the `http` launch profile.
3. Press **F5** or choose **Debug → Start Debugging**.
4. Visual Studio launches the site at `http://localhost:5194`.

The launch profile has `launchBrowser` enabled, so the browser opens automatically.

## Project layout

- `Pages/Index.cshtml` — the homepage content and Razor entry point
- `Pages/Shared/_Layout.cshtml` — shared HTML head, asset links, page shell, and scripts section
- `Pages/Shared/_Header.cshtml` — reusable site header/navigation
- `Pages/Shared/_Footer.cshtml` — reusable site footer
- `wwwroot/css/` — base, component, responsive, and stylesheet entry-point files
- `wwwroot/js/data.js` — course, department, project, and discipline data
- `wwwroot/js/catalog-extension.js` — expands and normalizes the catalog to 200 projects across 10 disciplines
- `wwwroot/js/app.js` — navigation, rendering, routing, and interaction logic
- `wwwroot/images/` — optimized public images; keep private uploads outside `wwwroot`
- `Program.cs` — ASP.NET Core application startup
- `Properties/launchSettings.json` — Visual Studio run/debug profile

## Command-line preview

```powershell
dotnet run --launch-profile http
```

The project targets .NET 10 and builds successfully with the installed SDK.
