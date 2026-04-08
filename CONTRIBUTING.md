# Contributing to Modal Vertical AI Demo

Thank you for your interest in contributing! This project is meant to be a clean, educational demo of Modal's capabilities.

## How to Contribute

### Reporting Issues

- Check existing issues first
- Provide clear reproduction steps
- Include Modal version, Node version, Python version
- Share relevant logs or error messages

### Suggesting Enhancements

- Describe the use case clearly
- Explain how it helps others learn Modal
- Keep scope focused on the demo's educational purpose

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test the full pipeline (`npm run load && npm run rollouts && npm run finetune && npm run serve`)
5. Update README if needed
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Code Style

- **TypeScript**: Follow existing style, use Prettier
- **Python**: Follow PEP 8, use Black formatter
- **Comments**: Explain WHY, not just WHAT (this is educational code)
- **Keep it simple**: This is a demo, clarity over cleverness

### Testing Changes

Make sure to test:
- All four pipeline steps run successfully
- TypeScript compiles without errors
- Python functions deploy to Modal
- README instructions are accurate

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update this CONTRIBUTING.md if the process changes

## Questions?

Open an issue with the "question" label and we'll help you out!

---

**Thanks for helping make this demo better!** 🚀
