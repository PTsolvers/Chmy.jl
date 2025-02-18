import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  ignoreDeadLinks: true,
  vite: {
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },
  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"}
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav: [
      { text: "Home", link: "/" },
      { text: "Getting Started",
        items: [
          { text: "Introduction", link: "/getting_started/introduction" },
          { text: "Chmy.jl with MPI", link: "/getting_started/using_chmy_with_mpi" },
        ],
      },
      { text: "Concepts",
        items: [
          {text: "Architecture", link: "/concepts/architectures"},
          {text: "Grids", link: "/concepts/grids"},
          {text: "Fields", link: "/concepts/fields"},
          {text: "Boundary Conditions", link: "/concepts/bc"},
          {text: "Grid Operators", link: "/concepts/grid_operators"},
          {text: "Kernels", link: "/concepts/kernels"},
          {text: "Distributed", link: "/concepts/distributed"},
        ],
      },
      { text: "Examples",
        items: [
          { text: "Examples Overview", link: "/examples/overview" },
        ],
      },
      { text: "Library",
        items: [
          { text: "Modules", link: "/lib/modules" },
        ],
      },
      { text: "Developer Doc",
        items: [
          { text: "Running Tests", link: "/developer_doc/running_tests" },
          { text: "Workers", link: "/developer_doc/workers" },
        ],
      },
    ],
    sidebar: {
      "/getting_started/": [
        {
          text: "Getting Started",
          collapsed: false,
          items: [
            { text: "Introduction", link: "/getting_started/introduction" },
            { text: "Chmy.jl with MPI", link: "/getting_started/using_chmy_with_mpi" },
          ]
        }
      ],
      "/concepts/": [
        {
          text: "Concepts",
          collapsed: false,
          items: [
            {text: "Architecture", link: "/concepts/architectures"},
            {text: "Grids", link: "/concepts/grids"},
            {text: "Fields", link: "/concepts/fields"},
            {text: "Boundary Conditions", link: "/concepts/bc"},
            {text: "Grid Operators", link: "/concepts/grid_operators"},
            {text: "Kernels", link: "/concepts/kernels"},
            {text: "Distributed", link: "/concepts/distributed"},
         ]
        }
      ],
      "/examples/": [
        {
          text: "Examples",
          collapsed: false,
          items: [
            { text: "Examples Overview", link: "/examples/overview" },
          ]
        }
      ],
      "/lib/": [
        {
          text: "Library",
          collapsed: false,
          items: [
            { text: "Modules", link: "/lib/modules" },
          ]
        }
      ],
      "/developer_doc/": [
        {
          text: "Developer Doc",
          collapsed: false,
          items: [
            { text: "Running Tests", link: "/developer_doc/running_tests" },
            { text: "Workers", link: "/developer_doc/workers" },
          ]
        }
      ],
    },
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' },
      { icon: 'slack', link: 'https://julialang.org/slack/' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()} ⋅ The Chmy Development Team.`
    }
  }
})
