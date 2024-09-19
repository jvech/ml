include config.mk

# you can `export DEV_MODE=true` to compile the binaries with more warnings and debugging support
ifdef DEV_MODE
CFLAGS 	= -std=gnu11 -Wall -Wextra -g
LDFLAGS = #-fsanitize=address
else
CFLAGS 	= -std=gnu11 -Wall -O2
LDFLAGS =
endif

CC 		= clang
BIN 	= ml
OBJDIR 	= objs
SRC 	= $(wildcard src/*.c)
HEADERS = $(wildcard src/*.h)
OBJS 	= $(SRC:src/%.c=${OBJDIR}/%.o) 
DLIBS 	= -lm $(shell pkg-config --libs-only-l blas json-c)
.PHONY: clean all run

all: build

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir ${OBJDIR}

$(OBJDIR)/%.o: src/%.c $(HEADERS)
	${CC} -c -o $@ $< ${CFLAGS}

build: $(OBJS)
	${CC} ${DLIBS} -o ${BIN} ${OBJS} ${LDFLAGS}

install: all
	@# binary
	install -d $(BINPREFIX)
	install -m 755 ${BIN} $(BINPREFIX)/${BIN}
	@#man page
	install -d $(MANPREFIX)/man1
	install -m 644 doc/ml.1 $(MANPREFIX)/man1/ml.1

install_config:
	mkdir -p ~/.config/ml
	sed "s!utils/weights.bin!${CFGPREFIX}/ml.weights!" utils/settings.cfg > ${CFGPREFIX}/ml.cfg

uninstall:
	rm -v $(BINPREFIX)/${BIN}
	rm -v $(MANPREFIX)/man1/ml.1

man: build
	help2man -N ./ml -I doc/man.txt > doc/ml.1

run: build
	@./${BIN} train data/sample_data.json | tee data/train_history.txt
	@./${BIN} predict data/sample_data.json | jq -r '.[] | [values[] as $$val | $$val] | @tsv' > data/net_data.tsv
	@jq -r '.[] | [values[] as $$val | $$val] | @tsv' data/sample_data.json > data/sample_data.tsv
	@gnuplot utils/plot.gpi

test_%: src/%.c $(OBJDIR)
	$(shell sed -n 's/.*compile: clang/clang/;/clang/p' $<)

debug: build
	gdb --tui --args ./${BIN} train -c utils/settings.cfg data/xor.csv
	@#gdb -x utils/commands.gdb --tui --args ${BIN} predict data/sample_data.json

check_leaks: build
	setarch x86_64 -R valgrind --log-file=leaks.log \
			 --leak-check=full \
			 --show-leak-kinds=all \
			 --track-origins=yes \
		./${BIN} train -c utils/settings.cfg data/xor.csv

clean:
	@rm $(OBJDIR) -rv
